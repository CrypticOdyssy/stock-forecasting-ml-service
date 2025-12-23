"""
Kafka consumer - Input adapter for preprocessing completion events.
"""
import json
import logging
from typing import Optional
from aiokafka import AIOKafkaConsumer
from .message_handler import PreprocessingEventHandler

logger = logging.getLogger(__name__)


class ForecastingConsumer:
    """Kafka consumer that listens for preprocessing completion events"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        event_handler: PreprocessingEventHandler,
        topic: str = 'data.preprocessing.completed',
        group_id: str = 'forecasting-service-group'
    ):
        self.bootstrap_servers = bootstrap_servers
        self.event_handler = event_handler
        self.topic = topic
        self.group_id = group_id
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_running = False
    
    def _deserialize_message(self, message_bytes):
        """Safely deserialize JSON message"""
        try:
            return json.loads(message_bytes.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to deserialize message: {e}")
            logger.error(f"Raw message: {message_bytes}")
            return None
    
    async def start(self):
        """Start consuming messages from Kafka"""
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=self._deserialize_message,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            await self.consumer.start()
            self.is_running = True
            logger.info(f"Kafka consumer started on topic '{self.topic}'")
            
            async for message in self.consumer:
                try:
                    if message.value is None:
                        logger.warning("Skipping invalid message")
                        continue
                    
                    await self.event_handler.handle(message.value)
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            self.is_running = False
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the consumer gracefully"""
        if self.consumer:
            await self.consumer.stop()
            self.is_running = False
            logger.info("Kafka consumer stopped")
