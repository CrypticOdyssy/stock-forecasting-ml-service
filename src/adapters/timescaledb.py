"""
TimescaleDB repository adapter (Output Adapter).
Implements data persistence for time series and forecasts.
"""
import logging
from typing import Optional, List
from datetime import datetime
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.domain.models import Forecast, ModelMetrics, ForecastMethod, ModelStatus
from src.domain.repositories import TimeSeriesRepository

logger = logging.getLogger(__name__)


class TimescaleDBRepository(TimeSeriesRepository):
    """
    TimescaleDB adapter for time series data access.
    Uses SQL Alchemy for connection management.
    """
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize required tables if they don't exist"""
        with self.engine.connect() as conn:
            # Enable TimescaleDB extension
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                conn.commit()
            except:
                logger.info("TimescaleDB extension already exists or cannot be created")
            
            # Create forecasts table (without inline INDEX)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id SERIAL PRIMARY KEY,
                    series_id VARCHAR(255) NOT NULL,
                    method VARCHAR(50) NOT NULL,
                    horizon INT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    mape FLOAT,
                    rmse FLOAT,
                    metadata JSONB,
                    forecast_data JSONB NOT NULL
                )
            """))
            conn.commit()
            
            # Create indexes separately
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_series_method 
                ON forecasts (series_id, method)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON forecasts (created_at)
            """))
            conn.commit()
            
            # Create model metrics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id SERIAL PRIMARY KEY,
                    series_id VARCHAR(255) NOT NULL,
                    method VARCHAR(50) NOT NULL,
                    train_rmse FLOAT NOT NULL,
                    test_rmse FLOAT NOT NULL,
                    train_mape FLOAT NOT NULL,
                    test_mape FLOAT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    last_trained TIMESTAMP NOT NULL,
                    training_samples INT NOT NULL,
                    UNIQUE(series_id, method)
                )
            """))
            conn.commit()
            
            # Create index for model_metrics
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_model_series_method 
                ON model_metrics (series_id, method)
            """))
            conn.commit()

    
    async def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve time series from time_series_preprocessed table"""
        try:
            query = "SELECT timestamp, close as value FROM time_series_preprocessed WHERE series_id = :series_id"
            params = {"series_id": series_id}
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params["start_date"] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params["end_date"] = end_date
            
            query += " ORDER BY timestamp"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.fetchall()
            
            if not rows:
                logger.warning(f"No data found for series {series_id}")
                return pd.DataFrame(columns=['timestamp', 'value'])
            
            df = pd.DataFrame(rows, columns=['timestamp', 'value'])
            # Remove timezone for compatibility with Prophet
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            logger.info(f"Fetched a total of {len(df)} rows")
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving series {series_id}: {e}")
            raise

    
    async def save_forecast(self, forecast: Forecast) -> None:
        """Persist forecast to database"""
        try:
            forecast_data = [p.to_dict() for p in forecast.points]
            
            query = text("""
                INSERT INTO forecasts 
                (series_id, method, horizon, created_at, mape, rmse, metadata, forecast_data)
                VALUES (:series_id, :method, :horizon, :created_at, :mape, :rmse, :metadata, :forecast_data)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    "series_id": forecast.series_id,
                    "method": forecast.method.value,
                    "horizon": forecast.horizon,
                    "created_at": forecast.created_at,
                    "mape": forecast.mape,
                    "rmse": forecast.rmse,
                    "metadata": json.dumps(forecast.metadata),
                    "forecast_data": json.dumps(forecast_data),
                })
                conn.commit()
            
            logger.info(f"Saved forecast for {forecast.series_id} using {forecast.method.value}")
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            raise
    
    async def get_latest_forecast(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[Forecast]:
        """Retrieve latest forecast for a series"""
        try:
            query = text("""
                SELECT * FROM forecasts 
                WHERE series_id = :series_id AND method = :method
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    "series_id": series_id,
                    "method": method.value
                })
                row = result.fetchone()
            
            if not row:
                return None
            
            forecast_points = [
                ForecastPoint(**p) for p in json.loads(row.forecast_data)
            ]
            
            return Forecast(
                series_id=row.series_id,
                method=ForecastMethod(row.method),
                horizon=row.horizon,
                created_at=row.created_at,
                points=forecast_points,
                mape=row.mape,
                rmse=row.rmse,
                metadata=json.loads(row.metadata) if row.metadata else {}
            )
        except Exception as e:
            logger.error(f"Error retrieving latest forecast: {e}")
            raise


class ModelCacheRepository:
    """
    Cache for trained models using filesystem and database.
    Models are serialized and stored for quick loading.
    """
    
    def __init__(self, timescaledb_repo: TimescaleDBRepository, cache_dir: str = "./models"):
        self.timescaledb_repo = timescaledb_repo
        self.cache_dir = cache_dir
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    async def save_model(
        self,
        series_id: str,
        method: ForecastMethod,
        model_data: bytes,
        metrics: ModelMetrics
    ) -> None:
        """Save model bytes and metrics"""
        import os
        
        # Save model file
        model_path = os.path.join(
            self.cache_dir,
            f"{series_id}_{method.value}.pkl"
        )
        
        with open(model_path, 'wb') as f:
            f.write(model_data)
        
        logger.info(f"Saved model to {model_path}")
        
        # Save metrics to database
        await self.update_model_metrics(metrics)
    
    async def load_model(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[bytes]:
        """Load model bytes"""
        import os
        
        model_path = os.path.join(
            self.cache_dir,
            f"{series_id}_{method.value}.pkl"
        )
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            return f.read()
    
    async def get_model_metrics(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[ModelMetrics]:
        """Get model metrics from database"""
        try:
            query = text("""
                SELECT * FROM model_metrics 
                WHERE series_id = :series_id AND method = :method
            """)
            
            with self.timescaledb_repo.engine.connect() as conn:
                result = conn.execute(query, {
                    "series_id": series_id,
                    "method": method.value
                })
                row = result.fetchone()
            
            if not row:
                return None
            
            return ModelMetrics(
                method=ForecastMethod(row.method),
                series_id=row.series_id,
                train_rmse=row.train_rmse,
                test_rmse=row.test_rmse,
                train_mape=row.train_mape,
                test_mape=row.test_mape,
                status=ModelStatus(row.status),
                last_trained=row.last_trained,
                training_samples=row.training_samples
            )
        except Exception as e:
            logger.error(f"Error retrieving model metrics: {e}")
            return None
    
    async def update_model_metrics(self, metrics: ModelMetrics) -> None:
        """Update or insert model metrics"""
        try:
            query = text("""
                INSERT INTO model_metrics 
                (series_id, method, train_rmse, test_rmse, train_mape, test_mape, status, last_trained, training_samples)
                VALUES (:series_id, :method, :train_rmse, :test_rmse, :train_mape, :test_mape, :status, :last_trained, :training_samples)
                ON CONFLICT (series_id, method) DO UPDATE SET
                    train_rmse = EXCLUDED.train_rmse,
                    test_rmse = EXCLUDED.test_rmse,
                    train_mape = EXCLUDED.train_mape,
                    test_mape = EXCLUDED.test_mape,
                    status = EXCLUDED.status,
                    last_trained = EXCLUDED.last_trained,
                    training_samples = EXCLUDED.training_samples
            """)
            
            with self.timescaledb_repo.engine.connect() as conn:
                conn.execute(query, {
                    "series_id": metrics.series_id,
                    "method": metrics.method.value,
                    "train_rmse": metrics.train_rmse,
                    "test_rmse": metrics.test_rmse,
                    "train_mape": metrics.train_mape,
                    "test_mape": metrics.test_mape,
                    "status": metrics.status.value,
                    "last_trained": metrics.last_trained,
                    "training_samples": metrics.training_samples
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
            raise


# Import for convenience
from src.domain.models import ForecastPoint