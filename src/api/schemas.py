"""
FastAPI schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ForecastRequest(BaseModel):
    """Request to generate forecast"""
    series_id: str = Field(..., description="Time series identifier (e.g., 'AAPL')")
    method: str = Field(..., description="Forecasting method: 'lstm' or 'prophet'")
    horizon: int = Field(default=30, ge=1, le=365, description="Days to forecast ahead")


class CompareForecastRequest(BaseModel):
    """Request to compare forecasting methods"""
    series_id: str
    horizon: int = Field(default=30, ge=1, le=365)


class TrainModelRequest(BaseModel):
    """Request to train a model"""
    series_id: str
    method: str = Field(..., description="'lstm' or 'prophet'")


class ForecastPointResponse(BaseModel):
    """Single forecast point"""
    timestamp: datetime
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResponse(BaseModel):
    """Complete forecast response"""
    series_id: str
    method: str
    horizon: int
    created_at: datetime
    points: List[ForecastPointResponse]
    mape: Optional[float] = None
    rmse: Optional[float] = None
    metadata: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "series_id": "AAPL",
                "method": "lstm",
                "horizon": 30,
                "created_at": "2024-12-17T19:00:00",
                "points": [
                    {
                        "timestamp": "2024-12-18T00:00:00",
                        "value": 150.5,
                        "lower_bound": 135.45,
                        "upper_bound": 165.55
                    }
                ],
                "mape": 0.052,
                "rmse": 2.34,
                "metadata": {}
            }
        }


class ComparativeForecastResponse(BaseModel):
    """Comparison of two forecasting methods"""
    series_id: str
    horizon: int
    created_at: datetime
    lstm_forecast: ForecastResponse
    prophet_forecast: ForecastResponse
    recommendation: str
    analysis: Dict[str, Any]


class ModelMetricsResponse(BaseModel):
    """Model performance metrics"""
    method: str
    series_id: str
    train_rmse: float
    test_rmse: float
    train_mape: float
    test_mape: float
    status: str
    last_trained: datetime
    training_samples: int


class ModelStatusResponse(BaseModel):
    """Status of all models for a series"""
    series_id: str
    lstm_status: Optional[ModelMetricsResponse] = None
    prophet_status: Optional[ModelMetricsResponse] = None