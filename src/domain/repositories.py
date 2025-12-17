"""
Repository interfaces (Ports) for data access abstraction.
These define contracts that adapters must implement.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime
from domain.models import Forecast, ModelMetrics, ForecastMethod, ModelStatus


class TimeSeriesRepository(ABC):
    """Port: Abstract interface for time series data access"""
    
    @abstractmethod
    async def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve time series data"""
        pass
    
    @abstractmethod
    async def save_forecast(self, forecast: Forecast) -> None:
        """Persist forecast results"""
        pass
    
    @abstractmethod
    async def get_latest_forecast(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[Forecast]:
        """Retrieve latest forecast for a series"""
        pass


class ModelRepository(ABC):
    """Port: Abstract interface for model persistence"""
    
    @abstractmethod
    async def save_model(
        self,
        series_id: str,
        method: ForecastMethod,
        model_data: bytes,
        metrics: ModelMetrics
    ) -> None:
        """Save trained model"""
        pass
    
    @abstractmethod
    async def load_model(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[bytes]:
        """Load trained model"""
        pass
    
    @abstractmethod
    async def get_model_metrics(
        self,
        series_id: str,
        method: ForecastMethod
    ) -> Optional[ModelMetrics]:
        """Get model performance metrics"""
        pass
    
    @abstractmethod
    async def update_model_metrics(
        self,
        metrics: ModelMetrics
    ) -> None:
        """Update model metrics"""
        pass


class PreprocessingPort(ABC):
    """Port: Abstract interface for preprocessing service"""
    
    @abstractmethod
    async def validate_data(self, series_id: str) -> Dict[str, Any]:
        """Validate time series data quality"""
        pass
    
    @abstractmethod
    async def create_features(
        self,
        series_id: str,
        lag_features: List[int],
        rolling_window_sizes: List[int]
    ) -> pd.DataFrame:
        """Create engineered features"""
        pass