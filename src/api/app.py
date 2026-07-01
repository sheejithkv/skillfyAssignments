"""
FastAPI application for Wine Quality prediction service.
"""

import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import (
    WineQualityInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.predict import prediction_service
from src.api.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PREDICTION_COUNT,
    PREDICTION_FAILURE_COUNT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="FastAPI inference service for the MLOps Wine Quality project.",
    version="1.0.0",
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(latency)

    return response


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "healthy",
        "service": "wine-quality-api",
    }


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    try:
        return prediction_service.get_model_info()
    except Exception as exc:
        logger.exception("Failed to fetch model info")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(payload: WineQualityInput):
    try:
        input_data = payload.model_dump(by_alias=True)

        prediction = prediction_service.predict(input_data)

        PREDICTION_COUNT.inc()

        return {
            "prediction": prediction,
            "rounded_prediction": int(round(prediction)),
            "model_name": prediction_service.metadata.get(
                "model_type",
                "WineQualityModel",
            ),
            "model_source": prediction_service.model_source,
        }

    except Exception as exc:
        PREDICTION_FAILURE_COUNT.inc()

        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
