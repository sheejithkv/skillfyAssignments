"""
Custom exception classes and decorators for the MLOps Wine Quality Project.

Features
--------
- Base project exception
- Specialized exceptions
- Automatic exception logging
- Decorator for centralized exception handling
"""

from functools import wraps
from typing import Callable, Any

from src.utils.logger import logger


class MLOpsException(Exception):
    """
    Base exception for the project.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class DataValidationException(MLOpsException):
    """Raised when dataset validation fails."""
    pass


class DataIngestionException(MLOpsException):
    """Raised during data ingestion."""
    pass


class PreprocessingException(MLOpsException):
    """Raised during preprocessing."""
    pass


class FeatureEngineeringException(MLOpsException):
    """Raised during feature engineering."""
    pass


class TrainTestSplitException(MLOpsException):
    """Raised during train/test split."""
    pass


class ModelTrainingException(MLOpsException):
    """Raised when model training fails."""
    pass


class ModelEvaluationException(MLOpsException):
    """Raised when model evaluation fails."""
    pass


class ModelLoadingException(MLOpsException):
    """Raised when model loading fails."""
    pass


class ModelSavingException(MLOpsException):
    """Raised when model saving fails."""
    pass


class PredictionException(MLOpsException):
    """Raised during inference."""
    pass


class MLflowException(MLOpsException):
    """Raised for MLflow-related errors."""
    pass


class ConfigurationException(MLOpsException):
    """Raised when configuration is invalid."""
    pass


class PipelineException(MLOpsException):
    """Raised when the pipeline execution fails."""
    pass


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator to log exceptions consistently and re-raise them.

    Example
    -------
    @handle_exceptions
    def preprocess():
        ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)

        except MLOpsException:
            logger.exception("Project exception occurred.")
            raise

        except Exception as exc:
            logger.exception(
                f"Unexpected exception in {func.__name__}: {str(exc)}"
            )
            raise MLOpsException(
                f"{func.__name__} failed: {str(exc)}"
            ) from exc

    return wrapper
