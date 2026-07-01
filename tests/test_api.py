"""
Integration tests for the FastAPI application.
"""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


SAMPLE_PAYLOAD = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


def test_health_endpoint():
    """
    Verify health endpoint.
    """

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "wine-quality-api"


def test_model_info_endpoint():
    """
    Verify model information endpoint.
    """

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "feature_count" in data
    assert "features" in data

    assert isinstance(data["features"], list)


def test_prediction_endpoint():
    """
    Verify prediction endpoint.
    """

    response = client.post(
        "/predict",
        json=SAMPLE_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "rounded_prediction" in data
    assert "model_name" in data
    assert "model_source" in data

    assert isinstance(data["prediction"], float)
    assert isinstance(data["rounded_prediction"], int)


def test_metrics_endpoint():
    """
    Verify Prometheus metrics endpoint.
    """

    response = client.get("/metrics")

    assert response.status_code == 200

    assert "wine_quality_api_requests_total" in response.text
    assert "wine_quality_predictions_total" in response.text


def test_invalid_request_returns_422():
    """
    Invalid payload should return HTTP 422.
    """

    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_invalid_data_type_returns_422():
    """
    Invalid data types should be rejected.
    """

    invalid_payload = SAMPLE_PAYLOAD.copy()
    invalid_payload["alcohol"] = "invalid"

    response = client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422
