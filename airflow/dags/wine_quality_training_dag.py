"""
Airflow DAG for Wine Quality MLOps Pipeline.

Pipeline:
1. Preprocess data
2. Train baseline model
3. Run Optuna tuning
4. Evaluate final model
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/mlops-end2end"

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="wine_quality_training_pipeline",
    description="Wine Quality MLOps training and evaluation pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "wine-quality", "training"],
) as dag:

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=f"cd {PROJECT_DIR} && python -m src.data.preprocess",
    )

    train = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.train",
    )

    tune = BashOperator(
        task_id="optuna_tuning",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.optuna_tuner",
    )

    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_DIR} && python -m src.models.evaluate",
    )

    preprocess >> train >> tune >> evaluate
