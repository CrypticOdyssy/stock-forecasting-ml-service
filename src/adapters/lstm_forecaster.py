"""
LSTM Neural Network Forecasting Adapter.
Implements LSTM for capturing complex temporal dependencies.
"""
import logging
from datetime import datetime, timedelta
from typing import Tuple
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from adapters.base_forecaster import BaseForecaster
from domain.models import ModelMetrics, ForecastMethod, ModelStatus

logger = logging.getLogger(__name__)


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based forecasting adapter.
    Best for: Complex temporal patterns, non-linear data, volatility.
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    async def forecast(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
        horizon: int = 30,
    ) -> Tuple[pd.DataFrame, ModelMetrics]:
        """Generate LSTM forecast"""
        logger.info(f"Generating LSTM forecast for {series_id}, horizon={horizon}")
        
        # Prepare data - use close price as the target value
        # Prepare data - already has 'value' column from preprocessed table
        data = historical_data[['timestamp', 'value']].copy()
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Validate minimum data requirements
        min_required = self.lookback * 2 + 50
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data: need at least {min_required} points, "
                f"but only have {len(data)}"
            )
        
        # Split data
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Ensure test set is large enough
        if len(test_data) <= self.lookback:
            raise ValueError(
                f"Test set too small: need more than {self.lookback} points, "
                f"but only have {len(test_data)}"
            )
        
        # Scale data
        scaled_train = self.scaler.fit_transform(train_data[['value']])
        scaled_test = self.scaler.transform(test_data[['value']])
        
        # Create sequences
        X_train, y_train = self._create_sequences(scaled_train)
        X_test, y_test = self._create_sequences(scaled_test)
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Insufficient data to create sequences")
        
        # Build and train model
        if not self.is_trained:
            self.model = self._build_model()
            self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=16,
                validation_data=(X_test, y_test),
                verbose=0
            )
            self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        train_pred_unscaled = self.scaler.inverse_transform(train_pred)
        test_pred_unscaled = self.scaler.inverse_transform(test_pred)
        
        train_rmse = float(np.sqrt(mean_squared_error(
            train_data[['value']].values[self.lookback:],
            train_pred_unscaled
        )))
        test_rmse = float(np.sqrt(mean_squared_error(
            test_data[['value']].values[self.lookback:],
            test_pred_unscaled
        )))
        
        train_mape = float(mean_absolute_percentage_error(
            train_data[['value']].values[self.lookback:],
            train_pred_unscaled
        ))
        test_mape = float(mean_absolute_percentage_error(
            test_data[['value']].values[self.lookback:],
            test_pred_unscaled
        ))
        
        metrics = ModelMetrics(
            method=ForecastMethod.LSTM,
            series_id=series_id,
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_mape=train_mape,
            test_mape=test_mape,
            status=ModelStatus.TRAINED,
            last_trained=datetime.utcnow(),
            training_samples=len(train_data)
        )
        
        # Generate future forecast
        last_sequence = scaled_test[-self.lookback:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        last_timestamp = data['timestamp'].iloc[-1]
        
        for i in range(horizon):
            next_pred = self.model.predict(
                np.array([current_sequence]),
                verbose=0
            )[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.vstack([
                current_sequence[1:],
                [[next_pred]]
            ])
        
        # Unscale predictions
        future_predictions_unscaled = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': [
                last_timestamp + timedelta(days=i+1)
                for i in range(horizon)
            ],
            'forecast': future_predictions_unscaled.flatten(),
        })
        
        # Add confidence intervals (±10%)
        forecast_df['lower_bound'] = forecast_df['forecast'] * 0.9
        forecast_df['upper_bound'] = forecast_df['forecast'] * 1.1
        
        return forecast_df, metrics
    
    async def train(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
    ) -> ModelMetrics:
        """Train LSTM model"""
        logger.info(f"Training LSTM model for {series_id}")
        
        data = historical_data[['timestamp', 'value']].copy()
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        scaled_train = self.scaler.fit_transform(train_data[['value']])
        scaled_test = self.scaler.transform(test_data[['value']])
        
        X_train, y_train = self._create_sequences(scaled_train)
        X_test, y_test = self._create_sequences(scaled_test)
        
        self.model = self._build_model()
        self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0
        )
        self.is_trained = True
        
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        train_pred_unscaled = self.scaler.inverse_transform(train_pred)
        test_pred_unscaled = self.scaler.inverse_transform(test_pred)
        
        train_rmse = float(np.sqrt(mean_squared_error(
            train_data[['value']].values[self.lookback:],
            train_pred_unscaled
        )))
        test_rmse = float(np.sqrt(mean_squared_error(
            test_data[['value']].values[self.lookback:],
            test_pred_unscaled
        )))
        
        train_mape = float(mean_absolute_percentage_error(
            train_data[['value']].values[self.lookback:],
            train_pred_unscaled
        ))
        test_mape = float(mean_absolute_percentage_error(
            test_data[['value']].values[self.lookback:],
            test_pred_unscaled
        ))
        
        return ModelMetrics(
            method=ForecastMethod.LSTM,
            series_id=series_id,
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            train_mape=train_mape,
            test_mape=test_mape,
            status=ModelStatus.TRAINED,
            last_trained=datetime.utcnow(),
            training_samples=len(train_data)
        )
    
    async def get_model_bytes(self) -> bytes:
        """Serialize model"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'lookback': self.lookback
        }
        return pickle.dumps(model_dict)
    
    async def load_model_bytes(self, model_bytes: bytes) -> None:
        """Deserialize model"""
        model_dict = pickle.loads(model_bytes)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.lookback = model_dict['lookback']
        self.is_trained = True
    
    def _build_model(self) -> Sequential:
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(128, activation='relu', input_shape=(self.lookback, 1)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)