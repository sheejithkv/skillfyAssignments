"""
Prediction service for FastAPI inference.
"""

import json

import joblib
import mlflow
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    MODEL_FILE,
    MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"


class PredictionService:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.metadata = {}
        self.model_source = "local"

    def load_local_model(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Local model not found: {MODEL_FILE}")

        if not FEATURE_NAMES_FILE.exists():
            raise FileNotFoundError(f"Feature names file not found: {FEATURE_NAMES_FILE}")

        self.model = joblib.load(MODEL_FILE)

        with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as file:
            self.feature_names = json.load(file)

        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        self.model_source = "local"
        logger.info("Local model loaded from %s", MODEL_FILE)

    def load_registry_model(
        self,
        model_name: str = "wine-quality-model",
        alias: str = "champion",
    ):
        model_uri = f"models:/{model_name}@{alias}"

        mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))

        self.model = mlflow.pyfunc.load_model(model_uri)

        if FEATURE_NAMES_FILE.exists():
            with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as file:
                self.feature_names = json.load(file)

        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)

        self.model_source = model_uri
        logger.info("Registry model loaded from %s", model_uri)

    def load_model(self):
        try:
            self.load_registry_model()
        except Exception as exc:
            logger.warning("Registry model load failed: %s", exc)
            logger.warning("Falling back to local model")
            self.load_local_model()

    def predict(self, input_data: dict) -> float:
        if self.model is None:
            self.load_model()

        input_df = pd.DataFrame([input_data])

        input_df = input_df[self.feature_names]

        prediction = self.model.predict(input_df)[0]

        return float(prediction)

    def get_model_info(self) -> dict:
        if self.model is None:
            self.load_model()

        return {
            "model_name": self.metadata.get("model_type", "WineQualityModel"),
            "model_source": self.model_source,
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
        }


prediction_service = PredictionService()
