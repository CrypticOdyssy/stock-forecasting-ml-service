from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IEventPublisher(ABC):
    """Port for publishing domain events to message broker"""
    
    @abstractmethod
    async def publish_forecast_completed(
        self,
        series_id: str,
        job_id: str,
        method: str,
        horizon: int,
        forecast_points: int,
        metadata: Dict[str, Any]
    ):
        """Publish forecast completion event"""
        pass
    
    @abstractmethod
    async def publish_processing_failed(
        self,
        series_id: str,
        job_id: str,
        error: str,
        stage: str
    ):
        """Publish processing failure event"""
        pass
