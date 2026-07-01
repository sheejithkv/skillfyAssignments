"""
Pydantic schemas for FastAPI inference service.
"""

from typing import List

from pydantic import BaseModel, Field


class WineQualityInput(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity")
    volatile_acidity: float = Field(..., alias="volatile acidity")
    citric_acid: float = Field(..., alias="citric acid")
    residual_sugar: float = Field(..., alias="residual sugar")
    chlorides: float
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide")
    density: float
    ph: float = Field(..., alias="pH")
    sulphates: float
    alcohol: float

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    prediction: float
    rounded_prediction: int
    model_name: str
    model_source: str


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_source: str
    feature_count: int
    features: List[str]
