"""
Core business logic and use cases for forecasting service.
This is the domain layer - independent of frameworks and adapters.
"""
from typing import Optional, Tuple
from datetime import datetime
import logging
import pandas as pd
import numpy as np

from domain.models import (
    Forecast, ForecastPoint, ComparativeForecast,
    ModelMetrics, ForecastMethod, ModelStatus
)
from domain.repositories import (
    TimeSeriesRepository, ModelRepository, PreprocessingPort
)

logger = logging.getLogger(__name__)


class ForecastingService:
    """
    Core service implementing forecasting business logic.
    Coordinates between repositories and ML adapters.
    """
    
    def __init__(
        self,
        timeseries_repo: TimeSeriesRepository,
        model_repo: ModelRepository,
        preprocessing_port: PreprocessingPort,
        lstm_forecaster,  # ML adapter injected
        prophet_forecaster,  # ML adapter injected
    ):
        self.timeseries_repo = timeseries_repo
        self.model_repo = model_repo
        self.preprocessing_port = preprocessing_port
        self.lstm_forecaster = lstm_forecaster
        self.prophet_forecaster = prophet_forecaster
    
    async def forecast(
        self,
        series_id: str,
        method: ForecastMethod,
        horizon: int = 30,
    ) -> Forecast:
        """
        Generate forecast using specified method.
        
        Args:
            series_id: Identifier for the time series
            method: Forecasting method (LSTM or PROPHET)
            horizon: Number of steps to forecast ahead
            
        Returns:
            Forecast object with predictions
        """
        logger.info(f"Starting forecast for {series_id} using {method.value}")
        
        # Retrieve historical data
        historical_data = await self.timeseries_repo.get_series(series_id)
        
        if historical_data.empty:
            raise ValueError(f"No historical data found for series {series_id}")
        
        # Validate data quality
        validation = await self.preprocessing_port.validate_data(series_id)
        #logger.info(f"Data validation: {validation}")
        
        # Load or train model
        if method == ForecastMethod.LSTM:
            forecaster = self.lstm_forecaster
        elif method == ForecastMethod.PROPHET:
            forecaster = self.prophet_forecaster
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        # Generate forecast
        forecast_df, metrics = await forecaster.forecast(
            historical_data,
            series_id=series_id,
            horizon=horizon
        )
        
        # Convert to domain model
        forecast = self._dataframe_to_forecast(
            forecast_df,
            series_id,
            method,
            horizon,
            metrics
        )
        
        # Persist forecast
        await self.timeseries_repo.save_forecast(forecast)
        
        # Update model metrics
        await self.model_repo.update_model_metrics(metrics)
        
        logger.info(f"Forecast completed for {series_id}")
        return forecast
    
    async def compare_methods(
        self,
        series_id: str,
        horizon: int = 30,
    ) -> ComparativeForecast:
        """
        Compare LSTM and Prophet forecasts.
        
        Returns recommendation based on data characteristics.
        """
        logger.info(f"Comparing forecasting methods for {series_id}")
        
        # Get historical data
        historical_data = await self.timeseries_repo.get_series(series_id)
        
        if historical_data.empty:
            raise ValueError(f"No historical data found for series {series_id}")
        
        # Generate LSTM forecast
        lstm_df, lstm_metrics = await self.lstm_forecaster.forecast(
            historical_data,
            series_id=series_id,
            horizon=horizon
        )
        lstm_forecast = self._dataframe_to_forecast(
            lstm_df, series_id, ForecastMethod.LSTM, horizon, lstm_metrics
        )
        
        # Generate Prophet forecast
        prophet_df, prophet_metrics = await self.prophet_forecaster.forecast(
            historical_data,
            series_id=series_id,
            horizon=horizon
        )
        prophet_forecast = self._dataframe_to_forecast(
            prophet_df, series_id, ForecastMethod.PROPHET, horizon, prophet_metrics
        )
        
        # Analyze and recommend
        recommendation, analysis = self._analyze_forecasts(
            historical_data,
            lstm_forecast,
            prophet_forecast,
            lstm_metrics,
            prophet_metrics
        )
        
        return ComparativeForecast(
            series_id=series_id,
            horizon=horizon,
            created_at=datetime.utcnow(),
            lstm_forecast=lstm_forecast,
            prophet_forecast=prophet_forecast,
            recommendation=recommendation,
            analysis=analysis
        )
    
    async def train_model(
        self,
        series_id: str,
        method: ForecastMethod,
    ) -> ModelMetrics:
        """Train and cache model for a specific series"""
        logger.info(f"Training {method.value} model for {series_id}")
        
        historical_data = await self.timeseries_repo.get_series(series_id)
        
        if method == ForecastMethod.LSTM:
            forecaster = self.lstm_forecaster
        elif method == ForecastMethod.PROPHET:
            forecaster = self.prophet_forecaster
        else:
            raise ValueError(f"Unknown forecasting method: {method}")
        
        # Train model
        metrics = await forecaster.train(historical_data, series_id=series_id)
        
        # Save model and metrics
        model_data = await forecaster.get_model_bytes()
        await self.model_repo.save_model(
            series_id,
            method,
            model_data,
            metrics
        )
        
        logger.info(f"Model training completed with RMSE: {metrics.test_rmse}")
        return metrics
    
    async def get_model_status(
        self,
        series_id: str,
        method: ForecastMethod,
    ) -> Optional[ModelMetrics]:
        """Get status and metrics for a trained model"""
        return await self.model_repo.get_model_metrics(series_id, method)
    
    def _dataframe_to_forecast(
        self,
        forecast_df: pd.DataFrame,
        series_id: str,
        method: ForecastMethod,
        horizon: int,
        metrics: ModelMetrics,
    ) -> Forecast:
        """Convert forecast dataframe to domain model"""
        points = []
        
        for idx, row in forecast_df.iterrows():
            point = ForecastPoint(
                timestamp=row['timestamp'],
                value=float(row['forecast']),
                lower_bound=float(row.get('lower_bound')),
                upper_bound=float(row.get('upper_bound')),
            )
            points.append(point)
        
        return Forecast(
            series_id=series_id,
            method=method,
            horizon=horizon,
            created_at=datetime.utcnow(),
            points=points,
            mape=metrics.test_mape,
            rmse=metrics.test_rmse,
            metadata={
                "training_samples": metrics.training_samples,
                "last_trained": metrics.last_trained.isoformat(),
            }
        )
    
    def _analyze_forecasts(
        self,
        historical_data: pd.DataFrame,
        lstm_forecast: Forecast,
        prophet_forecast: Forecast,
        lstm_metrics: ModelMetrics,
        prophet_metrics: ModelMetrics,
    ) -> Tuple[str, dict]:
        """
        Analyze forecasts and provide recommendation.
        
        Returns: (recommendation, analysis_dict)
        """
        analysis = {
            "lstm_rmse": lstm_metrics.test_rmse,
            "prophet_rmse": prophet_metrics.test_rmse,
            "lstm_mape": lstm_metrics.test_mape,
            "prophet_mape": prophet_metrics.test_mape,
        }
        
        # Calculate seasonality from historical data
        seasonality_score = self._detect_seasonality(historical_data)
        analysis["seasonality_score"] = seasonality_score
        
        # Calculate volatility
        volatility = float(historical_data['value'].pct_change().std())
        analysis["volatility"] = volatility
        
        # Recommendation logic
        if lstm_metrics.test_rmse < prophet_metrics.test_rmse:
            lstm_better = True
            improvement = (
                (prophet_metrics.test_rmse - lstm_metrics.test_rmse) /
                prophet_metrics.test_rmse * 100
            )
        else:
            lstm_better = False
            improvement = (
                (lstm_metrics.test_rmse - prophet_metrics.test_rmse) /
                lstm_metrics.test_rmse * 100
            )
        
        analysis["method_advantage"] = "LSTM" if lstm_better else "Prophet"
        analysis["advantage_percentage"] = improvement
        
        if seasonality_score > 0.6 and not lstm_better:
            recommendation = (
                "Use Prophet - High seasonality detected and Prophet "
                f"outperforms LSTM by {improvement:.1f}%"
            )
        elif lstm_better:
            recommendation = (
                f"Use LSTM - Superior performance ({improvement:.1f}% better RMSE) "
                "and better at capturing complex patterns"
            )
        else:
            if volatility > 0.1:
                recommendation = (
                    "Use LSTM - High volatility detected, "
                    "LSTM better captures dynamic patterns"
                )
            else:
                recommendation = "Use Prophet - Simpler data, Prophet is more efficient"
        
        return recommendation, analysis
    
    def _detect_seasonality(self, data: pd.DataFrame) -> float:
        """
        Detect seasonality strength (0-1).
        Uses autocorrelation at seasonal lags.
        """
        try:
            if len(data) < 30:
                return 0.0
            
            values = data['value'].values
            # Check for 7-day and 30-day seasonality
            acf_7 = abs(np.corrcoef(values[:-7], values[7:])[0, 1])
            acf_30 = abs(np.corrcoef(values[:-30], values[30:])[0, 1])
            
            seasonality = max(acf_7, acf_30)
            return min(float(seasonality), 1.0)
        except:
            return 0.0