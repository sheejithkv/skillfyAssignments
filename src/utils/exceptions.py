"""
Custom exceptions for the MLOps Wine Quality Project.
"""

from functools import wraps

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLOpsException(Exception):
    """Base project exception."""


class DataValidationException(MLOpsException):
    pass


class DataIngestionException(MLOpsException):
    pass


class PreprocessingException(MLOpsException):
    pass


class TrainTestSplitException(MLOpsException):
    pass


class ModelTrainingException(MLOpsException):
    pass


class ModelEvaluationException(MLOpsException):
    pass


class PredictionException(MLOpsException):
    pass


class MLflowException(MLOpsException):
    pass


class PipelineException(MLOpsException):
    pass


def handle_exceptions(func):
    """
    Decorator for consistent exception logging.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MLOpsException:
            logger.exception("Project exception occurred in %s", func.__name__)
            raise
        except Exception as exc:
            logger.exception("Unexpected exception in %s", func.__name__)
            raise MLOpsException(str(exc)) from exc

    return wrapper
