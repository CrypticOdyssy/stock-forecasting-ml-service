"""
Handles preprocessing completion events and triggers forecasting.
"""
import logging
from typing import Dict, Any
from src.domain.service import ForecastingService
from src.domain.models import ForecastMethod
from src.domain.ports import IEventPublisher

from shared import SimpleJobTracker

logger = logging.getLogger(__name__)


class PreprocessingEventHandler:
    """Handles preprocessing completion events and triggers forecasting"""
    
    def __init__(
        self,
        forecasting_service: ForecastingService,
        event_publisher: IEventPublisher
    ):
        self.forecasting_service = forecasting_service
        self.event_publisher = event_publisher
    
    async def handle(self, event_data: Dict[str, Any]):
        """
        Process preprocessing completion event.
        
        Args:
            event_data: Event payload containing series_id, job_id, etc.
        """
        series_id = None
        job_id = None
        
        try:
            logger.info(f"Processing preprocessing completion event: {event_data.get('series_id')}")
            
            series_id = event_data.get('series_id')
            job_id = event_data.get('job_id')
            forecast_config = event_data.get('forecast_config', {})
            
            if not series_id or not job_id:
                raise ValueError("Missing required fields: series_id or job_id")
            
            # Mark forecasting as started
            SimpleJobTracker.update_status(
                job_id=job_id,
                series_id=series_id,
                status='running',
                stage='forecasting'
            )
            
            # Get forecast configuration
            method = ForecastMethod(forecast_config.get('method', 'lstm'))
            horizon = forecast_config.get('horizon', 30)
            
            # Execute forecasting (domain logic)
            logger.info(f"Starting forecast for {series_id} using {method.value}")
            forecast = await self.forecasting_service.forecast(
                series_id,
                method,
                horizon
            )

            SimpleJobTracker.update_status(
                job_id=job_id,
                series_id=series_id,
                status='completed',
                stage='forecasting',
                metadata={
                    'horizon': horizon
                }
            )
            
            # Publish success event
            await self.event_publisher.publish_forecast_completed(
                series_id=series_id,
                job_id=job_id,
                method=forecast.method.value,
                horizon=forecast.horizon,
                forecast_points=len(forecast.points),
                metadata={
                    'mape': forecast.mape,
                    'rmse': forecast.rmse,
                }
            )
            
            logger.info(f"Successfully forecasted series {series_id} for job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing preprocessing completion event: {e}", exc_info=True)
            
            if series_id and job_id:
                SimpleJobTracker.update_status(
                    job_id=job_id,
                    series_id=series_id,
                    status='failed',
                    stage='forecasting',
                    error_message=str(e)
                )
                await self.event_publisher.publish_processing_failed(
                    series_id=series_id,
                    job_id=job_id,
                    error=str(e),
                    stage='forecasting'
                )
