"""
Unit tests for model evaluation.
"""

import json
from pathlib import Path

from src.config import METRICS_DIR
from src.data.preprocess import preprocess
from src.models.train import train_model
from src.models.evaluate import evaluate_model


METRICS_FILE = METRICS_DIR / "metrics.json"


def test_evaluation_returns_metrics():
    """
    Evaluation should return a metrics dictionary.
    """

    preprocess()
    train_model()

    metrics = evaluate_model()

    assert isinstance(metrics, dict)

    expected_keys = {
        "mae",
        "mse",
        "rmse",
        "r2_score",
    }

    assert expected_keys.issubset(metrics.keys())


def test_metrics_file_created():
    """
    metrics.json should be created.
    """

    preprocess()
    train_model()
    evaluate_model()

    assert Path(METRICS_FILE).exists()


def test_metrics_file_contains_expected_keys():
    """
    metrics.json should contain all expected metrics.
    """

    preprocess()
    train_model()
    evaluate_model()

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics


def test_metric_values_are_numeric():
    """
    All metric values should be numeric.
    """

    preprocess()
    train_model()

    metrics = evaluate_model()

    for value in metrics.values():
        assert isinstance(value, (float, int))
