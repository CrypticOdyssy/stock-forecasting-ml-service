"""
Kafka producer - Output adapter for publishing events.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from aiokafka import AIOKafkaProducer
from src.domain.ports import IEventPublisher

logger = logging.getLogger(__name__)


class KafkaEventPublisher(IEventPublisher):
    """Kafka implementation of event publisher port"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        completed_topic: str = 'data.forecast.completed',
        failed_topic: str = 'data.processing.failed'
    ):
        self.bootstrap_servers = bootstrap_servers
        self.completed_topic = completed_topic
        self.failed_topic = failed_topic
        self.producer: Optional[AIOKafkaProducer] = None
    
    async def _get_producer(self) -> AIOKafkaProducer:
        """Lazy initialization of producer"""
        if self.producer is None:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip'
            )
            await self.producer.start()
            logger.info(f"Kafka producer started: {self.bootstrap_servers}")
        return self.producer
    
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
        event = {
            'event_type': 'forecast.completed',
            'timestamp': datetime.utcnow().isoformat(),
            'series_id': series_id,
            'job_id': job_id,
            'method': method,
            'horizon': horizon,
            'forecast_points': forecast_points,
            'metadata': metadata
        }
        
        try:
            producer = await self._get_producer()
            await producer.send(self.completed_topic, value=event)
            logger.info(
                f"Published forecast completed event - "
                f"Job: {job_id}, Series: {series_id}, Method: {method}"
            )
        except Exception as e:
            logger.error(f"Failed to publish completion event: {e}", exc_info=True)
            raise
    
    async def publish_processing_failed(
        self,
        series_id: str,
        job_id: str,
        error: str,
        stage: str
    ):
        """Publish processing failure event"""
        event = {
            'event_type': 'processing.failed',
            'timestamp': datetime.utcnow().isoformat(),
            'series_id': series_id,
            'job_id': job_id,
            'stage': stage,
            'error': error
        }
        
        try:
            producer = await self._get_producer()
            await producer.send(self.failed_topic, value=event)
            logger.error(
                f"Published processing failed event - "
                f"Job: {job_id}, Series: {series_id}, Stage: {stage}"
            )
        except Exception as e:
            logger.error(f"Failed to publish failure event: {e}", exc_info=True)
    
    async def close(self):
        """Close producer connection"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer closed")
