from src.data.preprocess import preprocess
from src.models.train import train_model
from src.models.evaluate import evaluate_model

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():

    logger.info("========== PIPELINE STARTED ==========")

    preprocess()

    train_model()

    metrics = evaluate_model()

    logger.info("========== PIPELINE COMPLETED ==========")

    return metrics


if __name__ == "__main__":
    run_pipeline()
