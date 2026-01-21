"""
Test cases for TimescaleDB repository adapter.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from datetime import datetime, timedelta
import pandas as pd
import json
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.adapters.timescaledb import (
    TimescaleDBRepository,
    ModelCacheRepository
)
from src.domain.models import (
    Forecast,
    ForecastPoint,
    ForecastMethod,
    ModelMetrics,
    ModelStatus
)


class TestTimescaleDBRepositoryInitialization:
    """Test repository initialization and table creation."""
    
    def test_initialization_creates_engine(self):
        """Test that initialization creates SQLAlchemy engine."""
        with patch('src.adapters.timescaledb.create_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            repo = TimescaleDBRepository("postgresql://user:pass@localhost/db")
            
            mock_engine.assert_called_once_with("postgresql://user:pass@localhost/db")
    
    def test_init_tables_handles_timescaledb_extension_error(self):
        """Test that TimescaleDB extension errors are handled gracefully."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [Exception("Extension error"), None, None, None, None, None]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            # Should not raise, just log
            repo = TimescaleDBRepository("postgresql://test")
            assert repo is not None


class TestGetSeries:
    """Test time series data retrieval."""
    
    @pytest.mark.anyio
    async def test_get_series_basic_query(self):
        """Test basic series retrieval without date filters."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        
        # Mock data
        mock_rows = [
            (datetime(2025, 1, 1), 100.0),
            (datetime(2025, 1, 2), 105.0),
            (datetime(2025, 1, 3), 102.0),
        ]
        mock_result.fetchall.return_value = mock_rows
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            df = await repo.get_series("AAPL")
            
            assert len(df) == 3
            assert list(df.columns) == ['timestamp', 'value']
            assert df['value'].iloc[0] == 100.0
    
    @pytest.mark.anyio
    async def test_get_series_with_date_range(self):
        """Test series retrieval with start and end date filters."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(datetime(2025, 1, 1), 100.0)]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)
            df = await repo.get_series("AAPL", start_date=start, end_date=end)
            
            # Verify query params were passed
            call_args = mock_conn.execute.call_args
            assert 'start_date' in str(call_args)
            assert 'end_date' in str(call_args)
    
    @pytest.mark.anyio
    async def test_get_series_no_data_found(self):
        """Test handling when no data exists for series."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            df = await repo.get_series("INVALID")
            
            assert len(df) == 0
            assert list(df.columns) == ['timestamp', 'value']
    
    @pytest.mark.anyio
    async def test_get_series_removes_timezone(self):
        """Test that timezone is removed for Prophet compatibility."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        
        # Mock timezone-aware datetime
        import pytz
        tz_aware_dt = datetime(2025, 1, 1, tzinfo=pytz.UTC)
        mock_result.fetchall.return_value = [(tz_aware_dt, 100.0)]
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            df = await repo.get_series("AAPL")
            
            # Verify timezone is removed
            assert df['timestamp'].iloc[0].tzinfo is None
    
    
@pytest.mark.anyio
class TestSaveForecast:
    """Test forecast persistence."""
    
    @pytest.mark.anyio
    async def test_save_forecast_success(self):
        """Test successful forecast save."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            mock_conn.reset_mock()  # Clear init calls
            
            forecast = Forecast(
                series_id="AAPL",
                method=ForecastMethod.PROPHET,
                horizon=30,
                created_at=datetime(2025, 1, 1),
                points=[
                    ForecastPoint(timestamp=datetime(2025, 1, 2), value=105.0),
                    ForecastPoint(timestamp=datetime(2025, 1, 3), value=106.0)
                ],
                mape=2.5,
                rmse=1.2,
                metadata={"test": "data"}
            )
            
            await repo.save_forecast(forecast)
            
            # Verify INSERT was called
            mock_conn.execute.assert_called()
            mock_conn.commit.assert_called()  # May be called multiple times including init
    
    @pytest.mark.anyio
    async def test_save_forecast_serializes_data_correctly(self):
        """Test that forecast data is correctly serialized to JSON."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            
            forecast = Forecast(
                series_id="AAPL",
                method=ForecastMethod.LSTM,
                horizon=7,
                created_at=datetime.now(),
                points=[ForecastPoint(timestamp=datetime(2025, 1, 2), value=100.0)],
                mape=1.5,
                rmse=0.8,
                metadata={"param": "value"}
            )
            
            await repo.save_forecast(forecast)
            
            # Verify JSON serialization in call
            call_args = mock_conn.execute.call_args
            params = call_args[0][1]
            assert params['series_id'] == "AAPL"
            assert params['method'] == "lstm"
            assert params['horizon'] == 7
    

@pytest.mark.anyio


class TestGetLatestForecast:
    """Test latest forecast retrieval."""
    
    @pytest.mark.anyio
    async def test_get_latest_forecast_success(self):
        """Test successful retrieval of latest forecast."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        
        forecast_data = [
            {"timestamp": "2025-01-02T00:00:00", "value": 105.0},
            {"timestamp": "2025-01-03T00:00:00", "value": 106.0}
        ]
        
        mock_row = MagicMock()
        mock_row.series_id = "AAPL"
        mock_row.method = "prophet"
        mock_row.horizon = 30
        mock_row.created_at = datetime(2025, 1, 1)
        mock_row.forecast_data = json.dumps(forecast_data)
        mock_row.mape = 2.5
        mock_row.rmse = 1.2
        mock_row.metadata = '{"test": "data"}'
        
        mock_result.fetchone.return_value = mock_row
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            
            forecast = await repo.get_latest_forecast("AAPL", ForecastMethod.PROPHET)
            
            assert forecast is not None
            assert forecast.series_id == "AAPL"
            assert forecast.method == ForecastMethod.PROPHET
            assert forecast.horizon == 30
            assert len(forecast.points) == 2
    
    @pytest.mark.anyio
    async def test_get_latest_forecast_not_found(self):
        """Test when no forecast exists."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        with patch('src.adapters.timescaledb.create_engine', return_value=mock_engine):
            repo = TimescaleDBRepository("postgresql://test")
            
            forecast = await repo.get_latest_forecast("INVALID", ForecastMethod.PROPHET)
            
            assert forecast is None
    
    @pytest.mark.anyio
    class TestModelCacheRepository:
        """Test model cache repository."""
        
        @pytest.fixture
        def mock_timescale_repo(self):
            """Create mock TimescaleDB repository."""
            mock_repo = MagicMock()
            mock_repo.engine = MagicMock()
            return mock_repo
        
        @pytest.mark.anyio
        async def test_initialization_creates_cache_directory(self, mock_timescale_repo):
            """Test that cache directory is created on init."""
            with patch('os.makedirs') as mock_makedirs:
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./test_models")
                
                mock_makedirs.assert_called_once_with("./test_models", exist_ok=True)
        
        @pytest.mark.anyio
        async def test_save_model_writes_file(self, mock_timescale_repo):
            """Test that model bytes are saved to file."""
            with patch('builtins.open', mock_open()) as mock_file, \
                patch('os.makedirs'):
                
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./models")
                
                model_data = b"model_bytes_here"
                metrics = ModelMetrics(
                    method=ForecastMethod.PROPHET,
                    series_id="AAPL",
                    train_rmse=0.5,
                    test_rmse=0.6,
                    train_mape=1.2,
                    test_mape=1.3,
                    status=ModelStatus.TRAINED,
                    last_trained=datetime.now(),
                    training_samples=1000
                )
                
                # Mock the update_model_metrics method
                with patch.object(cache_repo, 'update_model_metrics', new_callable=AsyncMock):
                    await cache_repo.save_model("AAPL", ForecastMethod.PROPHET, model_data, metrics)
                
                # Verify file was written
                mock_file.assert_called_once_with('./models/AAPL_prophet.pkl', 'wb')
        
        @pytest.mark.anyio
        async def test_save_model_updates_metrics(self, mock_timescale_repo):
            """Test that metrics are updated when saving model."""
            with patch('builtins.open', mock_open()), \
                patch('os.makedirs'):
                
                mock_conn = MagicMock()
                mock_timescale_repo.engine.connect.return_value.__enter__.return_value = mock_conn
                
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./models")
                
                metrics = ModelMetrics(
                    method=ForecastMethod.LSTM,
                    series_id="TSLA",
                    train_rmse=0.8,
                    test_rmse=0.9,
                    train_mape=2.1,
                    test_mape=2.2,
                    status=ModelStatus.TRAINED,
                    last_trained=datetime.now(),
                    training_samples=500
                )
                
                await cache_repo.save_model("TSLA", ForecastMethod.LSTM, b"data", metrics)
                
                # Verify database update was called
                mock_conn.execute.assert_called()
                mock_conn.commit.assert_called()
        
        @pytest.mark.anyio
        async def test_load_model_reads_file(self, mock_timescale_repo):
            """Test that model is loaded from file."""
            model_bytes = b"saved_model_data"
            
            with patch('builtins.open', mock_open(read_data=model_bytes)), \
                patch('os.path.exists', return_value=True), \
                patch('os.makedirs'):
                
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./models")
                
                loaded_data = await cache_repo.load_model("AAPL", ForecastMethod.PROPHET)
                
                assert loaded_data == model_bytes
        
        @pytest.mark.anyio
        async def test_load_model_file_not_found(self, mock_timescale_repo):
            """Test handling when model file doesn't exist."""
            with patch('os.path.exists', return_value=False), \
                patch('os.makedirs'):
                
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./models")
                
                loaded_data = await cache_repo.load_model("INVALID", ForecastMethod.PROPHET)
                
                assert loaded_data is None
        
        @pytest.mark.anyio
        async def test_get_model_metrics_success(self, mock_timescale_repo):
            """Test successful retrieval of model metrics."""
            mock_conn = MagicMock()
            mock_result = MagicMock()
            
            mock_row = MagicMock()
            mock_row.method = "prophet"
            mock_row.series_id = "AAPL"
            mock_row.train_rmse = 0.5
            mock_row.test_rmse = 0.6
            mock_row.train_mape = 1.2
            mock_row.test_mape = 1.3
            mock_row.status = "trained"
            mock_row.last_trained = datetime(2025, 1, 1)
            mock_row.training_samples = 1000
            
            mock_result.fetchone.return_value = mock_row
            mock_conn.execute.return_value = mock_result
            mock_timescale_repo.engine.connect.return_value.__enter__.return_value = mock_conn
            
            with patch('os.makedirs'):
                cache_repo = ModelCacheRepository(mock_timescale_repo, cache_dir="./models")
                
                metrics = await cache_repo.get_model_metrics("AAPL", ForecastMethod.PROPHET)
                
                assert metrics is not None
                assert metrics is not None
