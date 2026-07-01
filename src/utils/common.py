"""
Common utility functions for the MLOps Wine Quality Project.

This module contains reusable helper functions used across
the project.

Author: Sheejith
"""

from pathlib import Path
import json
import yaml
import joblib
import pandas as pd

from src.utils.logger import logger
from src.utils.exceptions import MLOpsException


# ==========================================================
# Directory Utilities
# ==========================================================

def create_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory : Path
        Directory path.
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
    except Exception as exc:
        logger.exception("Failed to create directory.")
        raise MLOpsException(str(exc))


# ==========================================================
# YAML Utilities
# ==========================================================

def read_yaml(path: Path) -> dict:
    """
    Read YAML file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        logger.info(f"Loaded YAML: {path}")
        return data

    except Exception as exc:
        logger.exception("Unable to read YAML.")
        raise MLOpsException(str(exc))


def write_yaml(path: Path, data: dict) -> None:
    """
    Write dictionary to YAML.
    """
    try:
        create_directory(path.parent)

        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                data,
                file,
                default_flow_style=False,
                sort_keys=False,
            )

        logger.info(f"YAML written: {path}")

    except Exception as exc:
        logger.exception("Unable to write YAML.")
        raise MLOpsException(str(exc))


# ==========================================================
# JSON Utilities
# ==========================================================

def read_json(path: Path) -> dict:
    """
    Read JSON file.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        logger.info(f"Loaded JSON: {path}")
        return data

    except Exception as exc:
        logger.exception("Unable to read JSON.")
        raise MLOpsException(str(exc))


def write_json(path: Path, data: dict) -> None:
    """
    Write dictionary to JSON.
    """
    try:
        create_directory(path.parent)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
            )

        logger.info(f"JSON written: {path}")

    except Exception as exc:
        logger.exception("Unable to write JSON.")
        raise MLOpsException(str(exc))


# ==========================================================
# CSV Utilities
# ==========================================================

def read_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV into DataFrame.
    """
    try:
        df = pd.read_csv(path)

        logger.info(f"CSV loaded: {path}")

        return df

    except Exception as exc:
        logger.exception("Unable to read CSV.")
        raise MLOpsException(str(exc))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to CSV.
    """
    try:
        create_directory(path.parent)

        df.to_csv(
            path,
            index=False,
        )

        logger.info(f"CSV saved: {path}")

    except Exception as exc:
        logger.exception("Unable to save CSV.")
        raise MLOpsException(str(exc))


# ==========================================================
# Joblib Utilities
# ==========================================================

def save_object(obj, path: Path) -> None:
    """
    Serialize Python object using joblib.
    """
    try:
        create_directory(path.parent)

        joblib.dump(obj, path)

        logger.info(f"Object saved: {path}")

    except Exception as exc:
        logger.exception("Unable to save object.")
        raise MLOpsException(str(exc))


def load_object(path: Path):
    """
    Load serialized object.
    """
    try:
        obj = joblib.load(path)

        logger.info(f"Object loaded: {path}")

        return obj

    except Exception as exc:
        logger.exception("Unable to load object.")
        raise MLOpsException(str(exc))


# ==========================================================
# Miscellaneous Utilities
# ==========================================================

def file_exists(path: Path) -> bool:
    """
    Check if file exists.
    """
    return Path(path).exists()


def ensure_file(path: Path) -> None:
    """
    Raise an exception if file does not exist.
    """
    if not Path(path).exists():
        raise MLOpsException(f"Required file not found: {path}")


def print_dataframe_info(df: pd.DataFrame) -> None:
    """
    Log basic DataFrame information.
    """
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing Values:\n{df.isnull().sum()}")
    logger.info(f"Duplicate Rows: {df.duplicated().sum()}")