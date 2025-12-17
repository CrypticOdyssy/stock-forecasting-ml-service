"""
Prophet Forecasting Adapter.
Statistical method for time series with strong seasonality.
"""
import logging
from datetime import datetime, timedelta
from typing import Tuple
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet

from adapters.base_forecaster import BaseForecaster
from src.domain.models import ModelMetrics, ForecastMethod, ModelStatus

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """
    Prophet-based forecasting adapter.
    Best for: Seasonal patterns, simple trends, minimal preprocessing.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    async def forecast(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
        horizon: int = 30,
    ) -> Tuple[pd.DataFrame, ModelMetrics]:
        """Generate Prophet forecast"""
        logger.info(f"Generating Prophet forecast for {series_id}, horizon={horizon}")
        
        # Prepare data for Prophet
        df_prophet = historical_data[['timestamp', 'value']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
        
        # Split data
        train_size = int(len(df_prophet) * 0.8)
        df_train = df_prophet[:train_size]
        df_test = df_prophet[train_size:]
        
        # Train Prophet model
        if not self.is_trained:
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.80,
                seasonality_mode='additive'
            )
            self.model.fit(df_train)
            self.is_trained = True
        
        # Make predictions on train and test
        train_forecast = self.model.predict(df_train[['ds']])
        test_forecast = self.model.predict(df_test[['ds']])
        
        # Calculate metrics
        train_actual = df_train['y'].values
        train_pred = train_forecast['yhat'].values
        
        test_actual = df_test['y'].values
        test_pred = test_forecast['yhat'].values
        
        train_rmse = float(np.sqrt(mean_squared_error(train_actual, train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(test_actual, test_pred)))
        
        train_mape = float(mean_absolute_percentage_error(train_actual, train_pred))
        test_mape = float(mean_absolute_percentage_error(test_actual, test_pred))
        
        metrics = ModelMetrics(
            method=ForecastMethod.PROPHET,
            series_id=series_id,
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_mape=train_mape,
            test_mape=test_mape,
            status=ModelStatus.TRAINED,
            last_trained=datetime.utcnow(),
            training_samples=len(df_train)
        )
        
        # Generate future forecast
        future_dates = pd.DataFrame({
            'ds': [
                df_prophet['ds'].max() + timedelta(days=i+1)
                for i in range(horizon)
            ]
        })
        
        forecast = self.model.predict(future_dates)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': forecast['ds'],
            'forecast': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper'],
        })
        
        return forecast_df, metrics
    
    async def train(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
    ) -> ModelMetrics:
        """Train Prophet model"""
        logger.info(f"Training Prophet model for {series_id}")
        
        df_prophet = historical_data[['timestamp', 'value']].copy()
        df_prophet.columns = ['ds', 'y']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
        
        train_size = int(len(df_prophet) * 0.8)
        df_train = df_prophet[:train_size]
        df_test = df_prophet[train_size:]
        
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
            seasonality_mode='additive'
        )
        self.model.fit(df_train)
        self.is_trained = True
        
        train_forecast = self.model.predict(df_train[['ds']])
        test_forecast = self.model.predict(df_test[['ds']])
        
        train_actual = df_train['y'].values
        train_pred = train_forecast['yhat'].values
        
        test_actual = df_test['y'].values
        test_pred = test_forecast['yhat'].values
        
        train_rmse = float(np.sqrt(mean_squared_error(train_actual, train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(test_actual, test_pred)))
        
        train_mape = float(mean_absolute_percentage_error(train_actual, train_pred))
        test_mape = float(mean_absolute_percentage_error(test_actual, test_pred))
        
        return ModelMetrics(
            method=ForecastMethod.PROPHET,
            series_id=series_id,
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_mape=train_mape,
            test_mape=test_mape,
            status=ModelStatus.TRAINED,
            last_trained=datetime.utcnow(),
            training_samples=len(df_train)
        )
    
    async def get_model_bytes(self) -> bytes:
        """Serialize Prophet model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        model_dict = {
            'model': self.model,
        }
        return pickle.dumps(model_dict)
    
    async def load_model_bytes(self, model_bytes: bytes) -> None:
        """Deserialize Prophet model"""
        model_dict = pickle.loads(model_bytes)
        self.model = model_dict['model']
        self.is_trained = True