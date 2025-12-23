"""
Dependency injection container for forecasting service.
"""
import os
import logging
from typing import Optional

from domain.service import ForecastingService
from domain.ports import IEventPublisher
from adapters.output.kafka import KafkaEventPublisher
from adapters.input.kafka import ForecastingConsumer, PreprocessingEventHandler

logger = logging.getLogger(__name__)


class ApplicationContainer:
    """Container managing application dependencies"""
    
    def __init__(self):
        self._event_publisher: Optional[IEventPublisher] = None
        self._forecasting_service: Optional[ForecastingService] = None
        self._kafka_consumer: Optional[ForecastingConsumer] = None
        self._bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    
    def get_event_publisher(self) -> IEventPublisher:
        """Get or create Kafka event publisher"""
        if self._event_publisher is None:
            self._event_publisher = KafkaEventPublisher(self._bootstrap_servers)
            logger.info("Kafka event publisher initialized")
        return self._event_publisher
    
    def get_forecasting_service(self) -> ForecastingService:
        """Get or create forecasting service"""
        if self._forecasting_service is None:
            from src.api.dependencies import get_service
            self._forecasting_service = get_service()
            logger.info("Forecasting service initialized")
        return self._forecasting_service
    
    def get_kafka_consumer(self) -> ForecastingConsumer:
        """Get or create Kafka consumer with wired dependencies"""
        if self._kafka_consumer is None:
            service = self.get_forecasting_service()
            publisher = self.get_event_publisher()
            
            event_handler = PreprocessingEventHandler(service, publisher)
            self._kafka_consumer = ForecastingConsumer(
                self._bootstrap_servers,
                event_handler
            )
            logger.info("Kafka consumer initialized")
        return self._kafka_consumer
    
    async def shutdown(self):
        """Clean up resources"""
        if self._kafka_consumer:
            await self._kafka_consumer.stop()
        if self._event_publisher:
            await self._event_publisher.close()
        logger.info("Application container shut down")


_container: Optional[ApplicationContainer] = None


def get_container() -> ApplicationContainer:
    """Get the application container singleton"""
    global _container
    if _container is None:
        _container = ApplicationContainer()
    return _container
