"""
Input adapters for Kafka event consumption.
"""
from .consumer import ForecastingConsumer
from .message_handler import PreprocessingEventHandler

__all__ = ['ForecastingConsumer', 'PreprocessingEventHandler']
