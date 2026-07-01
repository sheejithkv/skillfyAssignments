"""
Unit tests for model training.
"""

from pathlib import Path

from src.config import MODEL_FILE
from src.data.preprocess import preprocess
from src.models.train import train_model


def test_train_model_creates_model_file():
    preprocess()

    model = train_model()

    assert model is not None
    assert Path(MODEL_FILE).exists()


def test_trained_model_has_predict_method():
    preprocess()

    model = train_model()

    assert hasattr(model, "predict")
