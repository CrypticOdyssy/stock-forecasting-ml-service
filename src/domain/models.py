"""
Domain models and value objects for forecasting service.
Represents core business concepts independent of any framework or storage.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class ForecastMethod(str, Enum):
    """Supported forecasting methods"""
    LSTM = "lstm"
    PROPHET = "prophet"


class ModelStatus(str, Enum):
    """Model training status"""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


@dataclass
class ForecastPoint:
    """Single forecast point with confidence interval"""
    timestamp: datetime
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass
class Forecast:
    """Complete forecast result"""
    series_id: str
    method: ForecastMethod
    horizon: int
    created_at: datetime
    points: List[ForecastPoint]
    mape: Optional[float] = None
    rmse: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "method": self.method.value,
            "horizon": self.horizon,
            "created_at": self.created_at.isoformat(),
            "points": [p.to_dict() for p in self.points],
            "mape": self.mape,
            "rmse": self.rmse,
            "metadata": self.metadata,
        }


@dataclass
class ComparativeForecast:
    """Comparison of multiple forecasting methods"""
    series_id: str
    horizon: int
    created_at: datetime
    lstm_forecast: Forecast
    prophet_forecast: Forecast
    recommendation: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "horizon": self.horizon,
            "created_at": self.created_at.isoformat(),
            "lstm_forecast": self.lstm_forecast.to_dict(),
            "prophet_forecast": self.prophet_forecast.to_dict(),
            "recommendation": self.recommendation,
            "analysis": self.analysis,
        }


@dataclass
class ModelMetrics:
    """Performance metrics for trained models"""
    method: ForecastMethod
    series_id: str
    train_rmse: float
    test_rmse: float
    train_mape: float
    test_mape: float
    status: ModelStatus
    last_trained: datetime
    training_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "series_id": self.series_id,
            "train_rmse": self.train_rmse,
            "test_rmse": self.test_rmse,
            "train_mape": self.train_mape,
            "test_mape": self.test_mape,
            "status": self.status.value,
            "last_trained": self.last_trained.isoformat(),
            "training_samples": self.training_samples,
        }