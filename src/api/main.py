"""
FastAPI application - REST API entry point with Kafka integration.
"""
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    CompareForecastRequest,
    ComparativeForecastResponse,
    TrainModelRequest,
    ModelStatusResponse,
    ModelMetricsResponse,
)
from api.dependencies import get_service, get_preprocessing_client
from domain.service import ForecastingService
from domain.models import ForecastMethod
from application.container import get_container

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for startup/shutdown with Kafka integration"""
    logger.info("Forecasting service starting...")
    container = get_container()
    
    try:
        # Start Kafka consumer in background
        consumer = container.get_kafka_consumer()
        asyncio.create_task(consumer.start())
        logger.info("Kafka consumer started successfully")
    except Exception as e:
        logger.error(f"Failed to start Kafka components: {e}", exc_info=True)
    
    yield
    
    logger.info("Forecasting service shutting down...")
    try:
        await container.shutdown()
        
        # Clean up preprocessing client
        preprocessing_client = get_preprocessing_client()
        await preprocessing_client.close()
        logger.info("All resources closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


app = FastAPI(
    title="Stock Forecasting Service",
    description="Hexagonal architecture with event-driven capabilities",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    container = get_container()
    consumer = container.get_kafka_consumer()
    
    return {
        "service": "Stock Forecasting Service",
        "status": "running",
        "version": "1.0.0",
        "methods": ["lstm", "prophet"],
        "kafka_consumer_running": consumer.is_running if consumer else False
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    container = get_container()
    consumer = container.get_kafka_consumer()
    
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "kafka_consumer": "running" if (consumer and consumer.is_running) else "stopped",
            "kafka_producer": "connected"
        }
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(
    request: ForecastRequest,
    service: ForecastingService = Depends(get_service)
):
    """
    Generate stock price forecast using specified method.

    **Methods:**
    - `lstm`: Deep learning approach, best for complex patterns and volatility
    - `prophet`: Statistical approach, best for seasonality and trends

    **Example:**
    ```json
    {
        "series_id": "AAPL",
        "method": "lstm",
        "horizon": 30
    }
    ```

    **Response:** Forecast with confidence intervals and performance metrics
    """
    try:
        logger.info(f"Forecast request: {request.series_id} using {request.method}")

        method = ForecastMethod(request.method)
        forecast = await service.forecast(
            request.series_id,
            method,
            request.horizon
        )

        return ForecastResponse(
            series_id=forecast.series_id,
            method=forecast.method.value,
            horizon=forecast.horizon,
            created_at=forecast.created_at,
            points=[
                {
                    "timestamp": p.timestamp,
                    "value": p.value,
                    "lower_bound": p.lower_bound,
                    "upper_bound": p.upper_bound,
                }
                for p in forecast.points
            ],
            mape=forecast.mape,
            rmse=forecast.rmse,
            metadata=forecast.metadata
        )
    except ValueError as e:
        logger.warning(e)
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/compare", response_model=ComparativeForecastResponse)
async def compare_methods(
    request: CompareForecastRequest,
    service: ForecastingService = Depends(get_service)
):
    """
    Compare LSTM and Prophet forecasts side-by-side.

    Returns both forecasts with intelligent recommendation based on:
    - Seasonality strength (Prophet preferred if high)
    - Volatility (LSTM preferred if high)
    - RMSE accuracy improvement metrics

    **Response:** Comparative analysis with recommendation and detailed metrics
    """
    try:
        logger.info(f"Compare methods request: {request.series_id}")

        comparison = await service.compare_methods(
            request.series_id,
            request.horizon
        )

        return ComparativeForecastResponse(
            series_id=comparison.series_id,
            horizon=comparison.horizon,
            created_at=comparison.created_at,
            lstm_forecast=ForecastResponse(
                series_id=comparison.lstm_forecast.series_id,
                method=comparison.lstm_forecast.method.value,
                horizon=comparison.lstm_forecast.horizon,
                created_at=comparison.lstm_forecast.created_at,
                points=[
                    {
                        "timestamp": p.timestamp,
                        "value": p.value,
                        "lower_bound": p.lower_bound,
                        "upper_bound": p.upper_bound,
                    }
                    for p in comparison.lstm_forecast.points
                ],
                mape=comparison.lstm_forecast.mape,
                rmse=comparison.lstm_forecast.rmse,
                metadata=comparison.lstm_forecast.metadata
            ),
            prophet_forecast=ForecastResponse(
                series_id=comparison.prophet_forecast.series_id,
                method=comparison.prophet_forecast.method.value,
                horizon=comparison.prophet_forecast.horizon,
                created_at=comparison.prophet_forecast.created_at,
                points=[
                    {
                        "timestamp": p.timestamp,
                        "value": p.value,
                        "lower_bound": p.lower_bound,
                        "upper_bound": p.upper_bound,
                    }
                    for p in comparison.prophet_forecast.points
                ],
                mape=comparison.prophet_forecast.mape,
                rmse=comparison.prophet_forecast.rmse,
                metadata=comparison.prophet_forecast.metadata
            ),
            recommendation=comparison.recommendation,
            analysis=comparison.analysis
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Compare error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/train", response_model=ModelMetricsResponse)
async def train_model(
    request: TrainModelRequest,
    service: ForecastingService = Depends(get_service)
):
    """
    Train and cache a forecasting model for a specific stock series.

    This endpoint trains the model on historical data and stores it for future use.
    Subsequent forecasts will use the cached model unless retraining is requested.

    **Parameters:**
    - `series_id`: Stock symbol (AAPL, GOOGL, etc.)
    - `method`: "lstm" or "prophet"

    **Response:** Training metrics (RMSE, MAPE, status, sample count)
    """
    try:
        logger.info(f"Train model request: {request.series_id} using {request.method}")

        method = ForecastMethod(request.method)
        metrics = await service.train_model(request.series_id, method)

        return ModelMetricsResponse(
            method=metrics.method.value,
            series_id=metrics.series_id,
            train_rmse=metrics.train_rmse,
            test_rmse=metrics.test_rmse,
            train_mape=metrics.train_mape,
            test_mape=metrics.test_mape,
            status=metrics.status.value,
            last_trained=metrics.last_trained,
            training_samples=metrics.training_samples
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status/{series_id}", response_model=ModelStatusResponse)
async def get_model_status(
    series_id: str,
    service: ForecastingService = Depends(get_service)
):
    """
    Get training status and performance metrics for both LSTM and Prophet models.

    Returns None for models that haven't been trained yet.
    This endpoint is useful for:
    - Checking if models are ready for forecasting
    - Comparing model performance before selecting method
    - Monitoring training history

    **Response:** Status of both models with metrics (or null if not trained)
    """
    try:
        logger.info(f"Status check for {series_id}")

        lstm_metrics = await service.get_model_status(
            series_id,
            ForecastMethod.LSTM
        )
        prophet_metrics = await service.get_model_status(
            series_id,
            ForecastMethod.PROPHET
        )

        return ModelStatusResponse(
            series_id=series_id,
            lstm_status=ModelMetricsResponse(
                method=lstm_metrics.method.value,
                series_id=lstm_metrics.series_id,
                train_rmse=lstm_metrics.train_rmse,
                test_rmse=lstm_metrics.test_rmse,
                train_mape=lstm_metrics.train_mape,
                test_mape=lstm_metrics.test_mape,
                status=lstm_metrics.status.value,
                last_trained=lstm_metrics.last_trained,
                training_samples=lstm_metrics.training_samples
            ) if lstm_metrics else None,
            prophet_status=ModelMetricsResponse(
                method=prophet_metrics.method.value,
                series_id=prophet_metrics.series_id,
                train_rmse=prophet_metrics.train_rmse,
                test_rmse=prophet_metrics.test_rmse,
                train_mape=prophet_metrics.train_mape,
                test_mape=prophet_metrics.test_mape,
                status=prophet_metrics.status.value,
                last_trained=prophet_metrics.last_trained,
                training_samples=prophet_metrics.training_samples
            ) if prophet_metrics else None
        )
    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)