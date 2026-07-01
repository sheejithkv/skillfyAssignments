"""
MLflow Model Registry utilities.
"""

import time

import mlflow
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(params: dict) -> str:
    mlflow_tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
    experiment_name = params.get("training", {}).get("experiment_name", "wine-quality")

    mlflow.set_tracking_uri(str(PROJECT_ROOT / mlflow_tracking_uri))
    mlflow.set_experiment(experiment_name)

    return str(PROJECT_ROOT / mlflow_tracking_uri)


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> int:
    """
    Register model from an MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID.

    model_name : str
        Registered model name.

    artifact_path : str
        MLflow artifact path where model was logged.

    Returns
    -------
    int
        Registered model version.
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info("Registering model from URI: %s", model_uri)

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    logger.info(
        "Model registered. Name=%s Version=%s",
        registered_model.name,
        registered_model.version,
    )

    return int(registered_model.version)


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str = "Staging",
) -> None:
    """
    Transition registered model to a stage.

    Valid stages: None, Staging, Production, Archived
    """

    client = MlflowClient()

    logger.info(
        "Transitioning model %s version %s to %s",
        model_name,
        version,
        stage,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=False,
    )

    logger.info("Model transitioned successfully.")


def set_model_alias(
    model_name: str,
    version: int,
    alias: str = "champion",
) -> None:
    """
    Set MLflow model alias.
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )

    logger.info(
        "Alias '%s' assigned to model %s version %s",
        alias,
        model_name,
        version,
    )


def wait_until_model_ready(
    model_name: str,
    version: int,
    timeout_seconds: int = 60,
) -> None:
    """
    Wait until model version is READY.
    """

    client = MlflowClient()
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        model_version = client.get_model_version(
            name=model_name,
            version=str(version),
        )

        if model_version.status == "READY":
            logger.info(
                "Model %s version %s is READY",
                model_name,
                version,
            )
            return

        logger.info(
            "Waiting for model version to be READY. Current status=%s",
            model_version.status,
        )

        time.sleep(2)

    raise TimeoutError(
        f"Model {model_name} version {version} not READY after {timeout_seconds}s"
    )


def load_model_from_registry(
    model_name: str,
    alias: str = "champion",
):
    """
    Load model using MLflow registry alias.
    """

    model_uri = f"models:/{model_name}@{alias}"

    logger.info("Loading model from registry URI: %s", model_uri)

    return mlflow.pyfunc.load_model(model_uri)
