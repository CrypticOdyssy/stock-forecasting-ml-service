"""
Preprocessing service client (External Adapter).
Communicates with preprocessing microservice using httpx.
"""
import logging
from typing import Dict, Any, List, Optional

import httpx
import pandas as pd

from src.domain.repositories import PreprocessingPort

logger = logging.getLogger(__name__)


class PreprocessingClient(PreprocessingPort):
    """
    HTTP client adapter for preprocessing service.
    Implements the PreprocessingPort interface.
    Uses httpx for both sync and async HTTP calls.
    """

    def __init__(self, base_url: str = "http://preprocessing:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Lazily create and reuse a single httpx AsyncClient.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                verify=True,
            )
        return self._client

    async def validate_data(self, series_id: str) -> Dict[str, Any]:
        """
        Call preprocessing service GET /validate/{series_id} endpoint.

        Expected response (ValidationResponse):
          {
            "total_points": int,
            "missing_values": int,
            "missing_percentage": float,
            "date_range": {"start": "...", "end": "..."},
            "value_stats": {"min": ..., "max": ..., "mean": ..., ...}
          }

        Returns: Dict with validation metadata, or error dict if call fails
        """
        url = f"{self.base_url}/validate/{series_id}"
        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Validation successful for %s: %d total points",
                series_id,
                data.get("total_points", 0)
            )
            return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(
                    "Series %s not found in preprocessing service (404)",
                    series_id
                )
                return {
                    "status": "not_found",
                    "series_id": series_id,
                    "error": "Series not found",
                }
            else:
                logger.error(
                    "HTTP error validating %s: %s %s",
                    series_id,
                    e.response.status_code,
                    e.response.text
                )
                return {
                    "status": "error",
                    "series_id": series_id,
                    "error": f"HTTP {e.response.status_code}",
                }

        except httpx.RequestError as e:
            logger.error(
                "Connection error calling preprocessing /validate for %s: %s",
                series_id,
                str(e)
            )
            return {
                "status": "error",
                "series_id": series_id,
                "error": f"Connection error: {str(e)}",
            }

        except Exception as e:
            logger.error(
                "Unexpected error validating %s: %s",
                series_id,
                str(e)
            )
            return {
                "status": "error",
                "series_id": series_id,
                "error": str(e),
            }

    async def create_features(
        self,
        series_id: str,
        lag_features: Optional[List[int]] = None,
        rolling_window_sizes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Call preprocessing service POST /features endpoint.

        Request (FeatureRequest):
          {
            "series_id": "sensor_123",
            "lag_features": [1, 7, 30],
            "rolling_window_sizes": [7, 14, 30]
          }

        Response (FeatureResponse):
          {
            "status": "success",
            "series_id": "sensor_123",
            "features": ["lag_1", "lag_7", "roll_7_mean", ...],
            "rows": 123
          }

        NOTE: The preprocessing API returns feature metadata (names, row count)
        but not the actual feature values. For a complete implementation,
        extend the preprocessing service to return feature data as well.

        For now, this returns an empty DataFrame with correct columns.

        Returns: DataFrame with feature columns (values populated externally)
        """
        lag_features = lag_features or []
        rolling_window_sizes = rolling_window_sizes or []

        url = f"{self.base_url}/features"
        payload = {
            "series_id": series_id,
            "lag_features": lag_features,
            "rolling_window_sizes": rolling_window_sizes,
        }

        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            logger.info(
                "Features created for %s: %d features, %d rows",
                series_id,
                len(data.get("features", [])),
                data.get("rows", 0)
            )

            # data is FeatureResponse:
            # {
            #   "status": "success",
            #   "series_id": "sensor_123",
            #   "features": ["lag_1", "roll_7_mean"],
            #   "rows": 123
            # }
            features = data.get("features") or []

            # Return DataFrame with correct structure.
            # Forecasting service can populate values as needed.
            return pd.DataFrame(columns=features)

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error creating features for %s: %s %s",
                series_id,
                e.response.status_code,
                e.response.text
            )
            return pd.DataFrame()

        except httpx.RequestError as e:
            logger.error(
                "Connection error calling preprocessing /features for %s: %s",
                series_id,
                str(e)
            )
            return pd.DataFrame()

        except Exception as e:
            logger.error(
                "Unexpected error creating features for %s: %s",
                series_id,
                str(e)
            )
            return pd.DataFrame()

    async def close(self) -> None:
        """
        Close the HTTP client connection.
        Call from FastAPI shutdown event for clean shutdown.
        """
        if self._client is not None:
            await self._client.aclose()
            logger.info("Preprocessing client closed")