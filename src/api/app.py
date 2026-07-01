"""
FastAPI application for Wine Quality prediction service.
"""

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    WineQualityInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.predict import prediction_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="FastAPI inference service for the MLOps Wine Quality project.",
    version="1.0.0",
)


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
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
