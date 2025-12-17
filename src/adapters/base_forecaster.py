"""
Base forecaster interface (Adapter).
Defines the contract for ML forecasting implementations.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
from domain.models import ModelMetrics


class BaseForecaster(ABC):
    """Base class for forecasting adapters"""
    
    @abstractmethod
    async def forecast(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
        horizon: int = 30,
    ) -> Tuple[pd.DataFrame, ModelMetrics]:
        """
        Generate forecast from historical data.
        
        Args:
            historical_data: DataFrame with 'timestamp' and 'value' columns
            series_id: Identifier for the series
            horizon: Steps ahead to forecast
            
        Returns:
            (forecast_df, metrics) where forecast_df has columns:
            - timestamp: forecast timestamp
            - forecast: predicted value
            - lower_bound: confidence interval lower
            - upper_bound: confidence interval upper
        """
        pass
    
    @abstractmethod
    async def train(
        self,
        historical_data: pd.DataFrame,
        series_id: str,
    ) -> ModelMetrics:
        """
        Train the forecasting model.
        
        Returns:
            ModelMetrics with performance scores
        """
        pass
    
    @abstractmethod
    async def get_model_bytes(self) -> bytes:
        """Serialize model to bytes for storage"""
        pass
    
    @abstractmethod
    async def load_model_bytes(self, model_bytes: bytes) -> None:
        """Deserialize model from bytes"""
        pass