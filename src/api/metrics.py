"""
Prometheus metrics for the Wine Quality FastAPI service.
"""

from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "wine_quality_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "http_status"],
)


REQUEST_LATENCY = Histogram(
    "wine_quality_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)


PREDICTION_COUNT = Counter(
    "wine_quality_predictions_total",
    "Total successful predictions",
)


PREDICTION_FAILURE_COUNT = Counter(
    "wine_quality_prediction_failures_total",
    "Total failed predictions",
)


MODEL_LOAD_FAILURE_COUNT = Counter(
    "wine_quality_model_load_failures_total",
    "Total model loading failures",
)
