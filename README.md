# Time Series Forecasting Service

A forecasting microservice built with hexagonal architecture, providing LSTM and Prophet-based predictions for time series data stored in TimescaleDB.

## Features

- **Multiple Forecasting Methods**: LSTM neural networks and Prophet
- **Model Comparison**: Side-by-side evaluation with intelligent recommendations
- **Model Training & Caching**: Train once, forecast multiple times
- **Model Status Tracking**: Monitor training metrics and model readiness
- **REST API**: FastAPI-based endpoints with automatic documentation
- **Time Series Storage**: Optimized TimescaleDB backend for efficient time series queries

## Architecture

Built using hexagonal (ports and adapters) architecture for maintainability and testability:
```
src/
├── api/ # REST API layer (adapters)
│ ├── main.py # FastAPI endpoints
│ ├── schemas.py # Request/response models
│ └── dependencies.py
├── domain/ # Business logic and models (core)
│ ├── service.py # Forecasting orchestration
│ └── models.py # Domain entities
├── adapters/ # Forecasting implementations & database
│ ├── lstm_forecaster.py
│ └── prophet_forecaster.py

```

### Key Components

- **Domain Layer**: Core business logic, model definitions, and service orchestration
- **API Layer**: FastAPI endpoints exposing forecasting functionality
- **Adapters**: Pluggable forecasting implementations (LSTM, Prophet) & Database repositories and data access

## Prerequisites

- Docker or Podman
- TimescaleDB instance
- Python 3.11+ (for local development)

## Quick Start

### Using Docker

Build the image:

```
docker build -t forecasting-service .
```


Run the container:

```
docker run -d
-p 8001:8001
-e DATABASE_URL="postgresql://user:pass@timescaledb:5432/tsdb"
--name forecasting
forecasting-service
```


### Using Podman
```
podman build -t forecasting-service .
podman run -d -p 8001:8001
-e DATABASE_URL="postgresql://user:pass@timescaledb:5432/tsdb"
forecasting-service

```

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string (required)
- `PREPROCESSING_SERVICE_URL`: URL to access preprocessing micro service
- `PYTHONPATH`: Module search path (set automatically in container)


### Health Check

**GET /** - Service status and available methods
```
curl http://localhost:8001/
```

```
Response:
{
    "service": "Stock Forecasting Service",
    "status": "running",
    "version": "1.0.0",
    "methods": ["lstm", "prophet"]
}
```


### Generate Forecast

**POST /forecast** - Generate forecast using specified method
```
    curl -X POST "http://localhost:8001/forecast"
    -H "Content-Type: application/json"
    -d '{
    "series_id": "AAPL",
    "method": "lstm",
    "horizon": 30
}'
```


**Methods:**
- `lstm`: Deep learning approach, best for complex patterns and volatility
- `prophet`: Statistical approach, best for seasonality and trends
```
Response (WIP):
{
    "series_id": "AAPL",
    "method": "lstm",
    "horizon": 30,
    "created_at": "2025-12-17T23:00:00Z",
    "points": [
        {
        "timestamp": "2025-12-18T00:00:00",
        "value": 185.42,
        "lower_bound": 166.88,
        "upper_bound": 203.96
        }
    ],
    "mape": 2.3,
    "rmse": 3.12,
    "metadata": {}
}
```


### Compare Methods

**POST /forecast/compare** - Compare LSTM and Prophet side-by-side

```
curl -X POST "http://localhost:8001/forecast/compare"
-H "Content-Type: application/json"
-d '{
"series_id": "GOOGL",
"horizon": 30
}'

```

Returns both forecasts with intelligent recommendation based on:
- Seasonality strength (Prophet preferred if high)
- Volatility (LSTM preferred if high)
- RMSE accuracy improvement metrics

Response includes:
- Complete LSTM forecast with metrics
- Complete Prophet forecast with metrics
- Recommendation on which method to use
- Analysis explaining the recommendation

### Train Model

**POST /models/train** - Train and cache a model

```
curl -X POST "http://localhost:8001/models/train"
-H "Content-Type: application/json"
-d '{
"series_id": "AAPL",
"method": "lstm"
}'
```


Trains the model on historical data and stores it for future use. Subsequent forecasts will use the cached model.

```
Response(WIP):
{
    "method": "lstm",
    "series_id": "AAPL",
    "train_rmse": 2.45,
    "test_rmse": 3.12,
    "train_mape": 1.8,
    "test_mape": 2.3,
    "status": "trained",
    "last_trained": "2025-12-17T23:00:00Z",
    "training_samples": 1200
}

```

### Check Model Status

**GET /models/status/{series_id}** - Get training status for both models

curl http://localhost:8001/models/status/SERIES_ID



Returns training metrics for both LSTM and Prophet models (or null if not trained).
```
Response:
{
    "series_id": "SERIES_ID",
    "lstm_status": {
        "method": "lstm",
        "series_id": "SERIES_ID",
        "train_rmse": 2.45,
        "test_rmse": 3.12,
        "train_mape": 1.8,
        "test_mape": 2.3,
        "status": "trained",
        "last_trained": "2025-12-17T23:00:00Z",
        "training_samples": 1200
    },
"prophet_status": null
}
```


## Database Schema

### Required Tables

#### time_series_preprocessed
-----


| Column | Type | Constraints |
| :-- | :-- | :-- |
| series_id | TEXT | NOT NULL, PRIMARY KEY |
| timestamp | TIMESTAMPTZ | NOT NULL, PRIMARY KEY |
| close | DOUBLE PRECISION |  |

#### forecasts
-----

| Column | Type | Constraints |
| :-- | :-- | :-- |
| id | SERIAL | PRIMARY KEY |
| series_id | VARCHAR(255) | NOT NULL |
| method | VARCHAR(50) | NOT NULL |
| horizon | INT | NOT NULL |
| created_at | TIMESTAMP | NOT NULL |
| mape | FLOAT |  |
| rmse | FLOAT |  |
| metadata | JSONB |  |
| forecast_data | JSONB | NOT NULL |

**Indexes:**

- `idx_series_method` on (series_id, method)
- `idx_created_at` on (created_at)


#### model_metrics
-----

| Column | Type | Constraints |
| :-- | :-- | :-- |
| id | SERIAL | PRIMARY KEY |
| series_id | VARCHAR(255) | NOT NULL |
| method | VARCHAR(50) | NOT NULL |
| train_rmse | FLOAT | NOT NULL |
| test_rmse | FLOAT | NOT NULL |
| train_mape | FLOAT | NOT NULL |
| test_mape | FLOAT | NOT NULL |
| status | VARCHAR(50) | NOT NULL |
| last_trained | TIMESTAMP | NOT NULL |
| training_samples | INT | NOT NULL |

-----

**Constraints:**

- UNIQUE (series_id, method)

**Indexes:**

- `idx_model_series_method` on (series_id, method)

## Disclaimer
This service was designed and implemented with assistance from AI-based coding and documentation tools to accelerate development and improve code quality.
