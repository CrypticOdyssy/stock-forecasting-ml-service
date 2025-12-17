"""
Dependency injection configuration.
Wires together domain services with adapters.
"""
import os
import logging
from functools import lru_cache

from domain.service import ForecastingService
from adapters.lstm_forecaster import LSTMForecaster
from adapters.prophet_forecaster import ProphetForecaster
from adapters.timescaledb import (
    TimescaleDBRepository,
    ModelCacheRepository
)
from adapters.preprocessing_client import PreprocessingClient

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_service() -> ForecastingService:
    """
    Dependency injection for ForecastingService.

    This function constructs the service with all its dependencies.
    Using lru_cache ensures singleton behavior across requests.

    Returns:
        ForecastingService: Fully initialized service with all adapters
    """

    # Configuration from environment
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://tsuser:ts_password@timescaledb:5432/timeseries"
    )
    preprocessing_url = os.getenv(
        "PREPROCESSING_SERVICE_URL",
        "http://preprocessing:8000"
    )

    logger.info(f"Initializing services with database: {database_url[:50]}...")
    logger.info(f"Preprocessing service URL: {preprocessing_url}")

    # Output adapters (Persistence)
    timeseries_repo = TimescaleDBRepository(database_url)
    model_cache_repo = ModelCacheRepository(timeseries_repo)

    # ML adapters
    lstm_forecaster = LSTMForecaster(lookback=60)
    prophet_forecaster = ProphetForecaster()

    # External adapters
    preprocessing_client = PreprocessingClient(preprocessing_url)

    # Core service with dependency injection
    service = ForecastingService(
        timeseries_repo=timeseries_repo,
        model_repo=model_cache_repo,
        preprocessing_port=preprocessing_client,
        lstm_forecaster=lstm_forecaster,
        prophet_forecaster=prophet_forecaster,
    )

    logger.info("ForecastingService initialized successfully")
    return service


def get_preprocessing_client() -> PreprocessingClient:
    """
    Get the preprocessing client singleton for manual lifecycle management.
    Useful for graceful shutdown.
    """
    service = get_service()
    return service.preprocessing_port