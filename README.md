# Wine Quality Prediction MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![MLflow](https://img.shields.io/badge/MLflow-Latest-blue)
![Optuna](https://img.shields.io/badge/Optuna-Latest-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success)

---

# Project Overview

This project implements an end-to-end MLOps pipeline for predicting wine quality using the Wine Quality dataset.

The objective is not only to train a machine learning model, but to demonstrate how a production-grade machine learning system is designed, built, deployed, monitored, and maintained.

The implementation follows modern MLOps best practices including:

- Data versioning
- Experiment tracking
- Hyperparameter optimization
- Model registry
- REST API serving
- Containerization
- Kubernetes deployment
- Monitoring
- Continuous Integration
- Automated testing

The entire workflow is reproducible and suitable for production deployment.

---

# Project Architecture



Wine Quality Dataset
│
▼
Data Ingestion
│
▼
Data Validation
│
▼
Preprocessing
│
▼
Train/Test Split
│
▼
Model Training
│
▼
Optuna Hyperparameter Tuning
│
▼
MLflow Experiment Tracking
│
▼
MLflow Model Registry
│
▼
FastAPI Prediction Service
│
▼
Docker Container
│
▼
Kubernetes Deployment
│
▼
Prometheus Monitoring
│
▼
Grafana Dashboard
│
▼
GitHub Actions CI/CD
│
▼
Airflow Pipeline



---

# Features

## Data Engineering

- Automated data ingestion
- Dataset validation
- Missing value verification
- Feature validation
- Train/Test split
- DVC ready

---

## Machine Learning

- Random Forest Regressor
- Hyperparameter tuning using Optuna
- Automatic model retraining
- Performance evaluation
- Feature persistence

---

## Experiment Tracking

- MLflow experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Model Registry integration

---

## Model Registry

- Automatic model registration
- Versioning
- Champion alias
- Registry-based inference

---

## API

FastAPI REST service with endpoints:



GET /health

GET /model-info

POST /predict

GET /metrics



---

## Deployment

- Docker
- Docker Compose
- Kubernetes
- ConfigMap
- Horizontal Pod Autoscaler
- Ingress

---

## Monitoring

- Prometheus Metrics
- Grafana Dashboard
- Request Counter
- Prediction Counter
- Request Latency
- Failure Metrics

---

## CI/CD

GitHub Actions

- Build
- Test
- Docker Build
- Pipeline Validation

---

## Testing

Pytest based testing including

- Data preprocessing
- Model training
- Model evaluation
- FastAPI endpoints
- Metrics endpoint

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.10 |
| ML | Scikit-Learn |
| Optimization | Optuna |
| Tracking | MLflow |
| API | FastAPI |
| Validation | Pydantic |
| Container | Docker |
| Orchestration | Kubernetes |
| Monitoring | Prometheus |
| Dashboard | Grafana |
| Workflow | Airflow |
| Version Control | Git |
| Data Versioning | DVC |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

# Project Structure



mlops-end2end/

├── airflow/
├── artifacts/
├── config/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── docker/
├── k8s/
├── logs/
├── mlruns/
├── models/
├── monitoring/
│   ├── grafana/
│   └── prometheus/
│
├── notebooks/
├── reports/
├── scripts/
├── src/
│
│   ├── api/
│   ├── data/
│   ├── models/
│   ├── pipeline/
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md



---

# Machine Learning Workflow



Dataset

↓

Validation

↓

Preprocessing

↓

Train/Test Split

↓

Random Forest Training

↓

Optuna Optimization

↓

Evaluation

↓

MLflow Registry

↓

FastAPI

↓

Deployment



---

# Current Project Status

| Module | Status |
|----------|---------|
| Project Setup | ✅ |
| Data Pipeline | ✅ |
| Model Training | ✅ |
| Model Evaluation | ✅ |
| Optuna | ✅ |
| MLflow | ✅ |
| Model Registry | ✅ |
| FastAPI | ✅ |
| Docker | ✅ |
| Kubernetes | ✅ |
| Prometheus | ✅ |
| Grafana | ✅ |
| Airflow | ✅ |
| GitHub Actions | ✅ |
| Testing | ✅ |

---

# Current Test Status



========================

16 Tests Passed

0 Failed

========================



---

This README documents an end-to-end production-ready MLOps implementation for the Wine Quality Prediction project.






**PART2**
---

# Dataset

The project uses the **Wine Quality Dataset**, a publicly available dataset containing physicochemical measurements of Portuguese red wine samples.

## Dataset Location


data/raw/winequality-red.csv


## Target Variable


quality


The model predicts the wine quality score based on eleven physicochemical properties.

## Input Features

| Feature |
|----------|
| fixed acidity |
| volatile acidity |
| citric acid |
| residual sugar |
| chlorides |
| free sulfur dioxide |
| total sulfur dioxide |
| density |
| pH |
| sulphates |
| alcohol |

---

# Installation

## Clone Repository

bash
git clone https://github.com/<your-github-username>/mlops-end2end.git

cd mlops-end2end


---

## Create Virtual Environment

Linux

bash
python3 -m venv venv

source venv/bin/activate


Windows

cmd
python -m venv venv

venv\Scripts\activate


---

## Install Dependencies

bash
pip install -r requirements.txt


---

# Verify Installation

bash
python -m compileall src


Expected result


Compilation successful


---

# Project Configuration

The project configuration is maintained in


params.yaml


Example

yaml
dataset:
  path: data/raw/winequality-red.csv

training:
  experiment_name: wine-quality
  registered_model_name: wine-quality-model

model:
  random_state: 42
  test_size: 0.2

optuna:
  n_trials: 20


---

# Running Individual Modules

## Data Ingestion

bash
python -m src.data.ingest


---

## Data Validation

bash
python -m src.data.validate


---

## Dataset Split

bash
python -m src.data.split


---

## Data Preprocessing

bash
python -m src.data.preprocess


Output


data/processed/train.csv

data/processed/test.csv


---

## Model Training

bash
python -m src.models.train


Output


artifacts/models/model.pkl

artifacts/models/model_metadata.json

artifacts/models/feature_names.json


---

## Hyperparameter Optimization

bash
python -m src.models.optuna_tuner


Output


artifacts/optuna/

best_trial.json

best_params.json

trials.csv


---

## Model Evaluation

bash
python -m src.models.evaluate


Output


artifacts/metrics/metrics.json

artifacts/reports/predictions.csv


---

# Running the Complete Pipeline

Instead of executing every module individually, execute the pipeline.

bash
python -m src.pipeline.pipeline


Pipeline Flow


Preprocess

↓

Train

↓

Evaluate


Future versions of the pipeline can include Optuna tuning and model registry promotion as optional stages depending on the deployment workflow.

---

# Project Artifacts

After execution, the project generates the following artifacts.


artifacts/

├── metrics/

├── models/

├── optuna/

└── reports/


---

# Logs

Application logs are written to


logs/mlops.log


The logs include

- Data preprocessing
- Training
- Evaluation
- Model loading
- API requests
- Prediction failures
- Registry operations

---

# MLflow

Start the MLflow UI

bash
mlflow ui


Open


http://localhost:5000


The MLflow dashboard contains

- Experiments
- Parameters
- Metrics
- Registered Models
- Model Versions
- Artifacts

---

# Model Registry

Registered model


wine-quality-model


Current workflow


Train Model

↓

Register Model

↓

Version Increment

↓

Champion Alias

↓

FastAPI loads Champion


Model versions are automatically created after every successful training run.

---

# Expected Outputs

Successful execution should generate


✓ Processed Dataset

✓ Trained Model

✓ Feature Metadata

✓ Evaluation Metrics

✓ Predictions

✓ MLflow Runs

✓ Registered Model

✓ Optuna Results





**PART3**

---

# FastAPI Inference Service

The project exposes a REST API using **FastAPI** for real-time wine quality prediction.

The API automatically loads the latest **Champion** model from the MLflow Model Registry.

If the registry is unavailable, it automatically falls back to the locally stored model.

---

# Running the API

Start the API server:

bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000


The server starts at:


http://localhost:8000


---

# Interactive API Documentation

FastAPI automatically generates interactive documentation.

## Swagger UI


http://localhost:8000/docs


Features

- Execute API requests directly
- View request schema
- View response schema
- Test endpoints without external tools

---

## ReDoc


http://localhost:8000/redoc


Provides a clean API reference generated from the OpenAPI specification.

---

# Available Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /health | Health check |
| GET | /model-info | Returns model information |
| POST | /predict | Predict wine quality |
| GET | /metrics | Prometheus metrics |

---

# Health Endpoint

Request

http
GET /health


Example

bash
curl http://localhost:8000/health


Response

json
{
    "status": "healthy",
    "service": "wine-quality-api"
}


---

# Model Information Endpoint

Request

http
GET /model-info


Example

bash
curl http://localhost:8000/model-info


Example Response

json
{
    "model_name": "RandomForestRegressor",
    "model_source": "models:/wine-quality-model@champion",
    "feature_count": 11,
    "features": [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ]
}


---

# Prediction Endpoint

Request

http
POST /predict


Example

bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
    "fixed acidity":7.4,
    "volatile acidity":0.70,
    "citric acid":0.00,
    "residual sugar":1.9,
    "chlorides":0.076,
    "free sulfur dioxide":11.0,
    "total sulfur dioxide":34.0,
    "density":0.9978,
    "pH":3.51,
    "sulphates":0.56,
    "alcohol":9.4
}'


Example Response

json
{
    "prediction": 5.04,
    "rounded_prediction": 5,
    "model_name": "RandomForestRegressor",
    "model_source": "models:/wine-quality-model@champion"
}


---

# Prometheus Metrics Endpoint

Request

http
GET /metrics


Example

bash
curl http://localhost:8000/metrics


Example Output


wine_quality_api_requests_total

wine_quality_predictions_total

wine_quality_prediction_failures_total

wine_quality_api_request_latency_seconds


These metrics are automatically scraped by Prometheus.

---

# MLflow Experiment Tracking

The project logs every training run to MLflow.

Each run records:

- Training parameters
- Hyperparameters
- Metrics
- Model artifacts
- Feature metadata
- Training timestamp

Launch the MLflow UI

bash
mlflow ui


Open


http://localhost:5000


---

# MLflow Model Registry

The trained model is automatically registered.

Current model


wine-quality-model


Every successful training run creates a new model version.

Example


Version 1

↓

Version 2

↓

Version 3

↓

Champion Alias


The FastAPI application loads the model using the alias:


models:/wine-quality-model@champion


This allows model upgrades without changing application code.

---

# Hyperparameter Optimization (Optuna)

Optuna is used to optimize the Random Forest model.

Optimized parameters include:

- Number of trees
- Maximum depth
- Minimum samples split
- Minimum samples leaf
- Bootstrap
- Maximum features

Run optimization

bash
python -m src.models.optuna_tuner


Generated artifacts


artifacts/optuna/

best_params.json

best_trial.json

trials.csv


The best model is automatically:

1. Retrained
2. Saved
3. Logged to MLflow
4. Registered
5. Assigned the champion alias

---

# Prediction Flow


Client

↓

FastAPI

↓

Load Champion Model

↓

Validate Request

↓

Predict

↓

Return JSON Response


---

# Error Handling

The API returns standard HTTP status codes.

| Status | Description |
|----------|-------------|
| 200 | Success |
| 422 | Invalid request payload |
| 500 | Internal server error |

Validation errors are handled automatically by FastAPI and Pydantic.

---

# API Testing

Run the complete API test suite:

bash
pytest tests/test_api.py -v


Expected result


6 passed


The API is fully covered by automated integration tests.




**PART4**
---

# Docker Deployment

The application is containerized using Docker to ensure a consistent runtime environment across development, testing, and production.

## Build Docker Image

bash
docker build -t wine-quality-api:1.0 .


Verify the image:

bash
docker images


Example:


REPOSITORY          TAG      IMAGE ID
wine-quality-api    1.0      xxxxxxxxxxxx


---

## Run Docker Container

bash
docker run -d \
  --name wine-quality-api \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/logs:/app/logs \
  wine-quality-api:1.0


Verify:

bash
docker ps


Check logs:

bash
docker logs wine-quality-api


---

## Verify API

Health

bash
curl http://localhost:8000/health


Model Information

bash
curl http://localhost:8000/model-info


Prediction

bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
"fixed acidity":7.4,
"volatile acidity":0.70,
"citric acid":0.00,
"residual sugar":1.9,
"chlorides":0.076,
"free sulfur dioxide":11,
"total sulfur dioxide":34,
"density":0.9978,
"pH":3.51,
"sulphates":0.56,
"alcohol":9.4
}'


---

# Kubernetes Deployment

The project includes Kubernetes manifests for deploying the API into a cluster.

Directory:


k8s/


Files:


deployment.yaml

service.yaml

configmap.yaml

hpa.yaml

ingress.yaml


---

## Deployment

bash
kubectl apply -f k8s/


Verify:

bash
kubectl get all


Pods

bash
kubectl get pods


Deployments

bash
kubectl get deployments


Services

bash
kubectl get svc


Ingress

bash
kubectl get ingress


---

# Horizontal Pod Autoscaler

The project includes an HPA.

Configuration


Minimum Replicas : 2

Maximum Replicas : 5

CPU Target : 70%


Verify

bash
kubectl get hpa


---

# Configuration Management

Application configuration is externalized through a Kubernetes ConfigMap.

Configured values include:

- MLflow tracking URI
- Model name
- Champion alias
- Log level
- Python runtime configuration

This enables environment-specific configuration without rebuilding the container image.

---

# Monitoring

The application exports Prometheus-compatible metrics.

Metrics endpoint


GET /metrics


Example

bash
curl http://localhost:8000/metrics


Available metrics include:


wine_quality_api_requests_total

wine_quality_predictions_total

wine_quality_prediction_failures_total

wine_quality_api_request_latency_seconds

wine_quality_model_load_failures_total


---

# Prometheus

Configuration file


monitoring/prometheus/prometheus.yml


Run Prometheus

bash
docker run -d \
--name prometheus \
-p 9090:9090 \
-v $(pwd)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
prom/prometheus


Open


http://localhost:9090


Example queries


wine_quality_api_requests_total

wine_quality_predictions_total

wine_quality_api_request_latency_seconds


---

# Grafana

Dashboard location


monitoring/grafana/dashboards/


Dashboard includes:

- Total Requests
- Request Rate
- Prediction Rate
- Average Latency
- P95 Latency
- Prediction Failures
- HTTP Status Codes

Typical workflow


Prometheus

↓

Grafana

↓

Dashboard Visualization


---

# Airflow Workflow

The training pipeline is orchestrated using Apache Airflow.

DAG location


airflow/dags/wine_quality_training_dag.py


Pipeline


Preprocess

↓

Train

↓

Optuna

↓

Evaluate


Run Airflow

bash
airflow standalone


Open


http://localhost:8080


The DAG supports scheduled execution as well as manual triggering from the Airflow UI.

---

# GitHub Actions

The project includes CI/CD workflows under:


.github/workflows/


Available workflows


ci.yml

docker.yml


CI pipeline performs:

- Source checkout
- Dependency installation
- Python compilation
- Dataset preprocessing
- Model training
- Model evaluation
- Automated test execution

Docker pipeline performs:

- Docker image build
- Image verification

---

# Deployment Workflow

The complete deployment workflow is:


Developer

↓

Git Push

↓

GitHub Actions

↓

Run Tests

↓

Build Docker Image

↓

Deploy Kubernetes

↓

Expose FastAPI

↓

Prometheus Scraping

↓

Grafana Dashboard


---

# Production Architecture


Dataset
    │
    ▼
Data Pipeline
    │
    ▼
Training
    │
    ▼
Optuna
    │
    ▼
MLflow
    │
    ▼
Model Registry
    │
    ▼
FastAPI
    │
    ▼
Docker
    │
    ▼
Kubernetes
    │
    ▼
Prometheus
    │
    ▼
Grafana
    │
    ▼
Airflow
    │
    ▼
GitHub Actions



**PART5**
---

# Automated Testing

The project includes a comprehensive automated test suite built using **pytest**.

Test categories:

- Data preprocessing
- Data validation
- Model training
- Model evaluation
- FastAPI endpoints
- Prometheus metrics

Project test structure


tests/

├── test_api.py

├── test_evaluation.py

├── test_preprocessing.py

└── test_training.py


---

## Execute All Tests

bash
python -m pytest tests -v


Example Result


=============================

16 passed

0 failed

=============================


---

## Generate Coverage Report

bash
pytest \
--cov=src \
--cov-report=term \
--cov-report=html


HTML report


htmlcov/index.html


---

# Model Performance

Current evaluation metrics obtained from the test dataset.

| Metric | Value |
|----------|-------:|
| MAE | 0.42 |
| MSE | 0.32 |
| RMSE | 0.57 |
| R² Score | 0.50 |

> These values depend on the train/test split and Optuna optimization results. Retraining may produce slightly different values due to the stochastic nature of the algorithm.

---

# Generated Project Artifacts

After executing the pipeline the following artifacts are generated.


artifacts/

├── metrics/

│   └── metrics.json

│

├── models/

│   ├── model.pkl

│   ├── model_metadata.json

│   └── feature_names.json

│

├── optuna/

│   ├── best_params.json

│   ├── best_trial.json

│   └── trials.csv

│

└── reports/

    └── predictions.csv


---

# Logging

Application logs are written to


logs/mlops.log


The log captures:

- Data ingestion
- Validation
- Preprocessing
- Model training
- Evaluation
- Optuna optimization
- MLflow operations
- Registry updates
- API requests
- Prediction errors

---

# Troubleshooting

## Dataset Not Found


FileNotFoundError


Verify


data/raw/winequality-red.csv


exists and is tracked by DVC if applicable.

---

## MLflow Cannot Load Model

Verify


mlruns/


exists and that the model has been registered.

Run

bash
python -m src.models.train


to register a new model version if required.

---

## API Returns HTTP 500

Check


logs/mlops.log


and verify:

- Model artifacts exist
- Feature metadata is present
- MLflow registry is accessible (or local fallback is available)

---

## Docker Container Exits

Inspect logs

bash
docker logs wine-quality-api


Verify required volumes are mounted and the application starts successfully.

---

## Kubernetes Pod Not Ready

Inspect resources

bash
kubectl describe pod <pod-name>

kubectl logs <pod-name>


Check:

- Readiness probe
- Liveness probe
- ConfigMap values
- Mounted volumes

---

# Future Enhancements

Potential improvements include:

- Model drift detection
- Data quality monitoring
- Feature store integration
- Canary deployments
- Blue/Green deployments
- Automated rollback
- Model explainability using SHAP
- Distributed training
- GPU support
- Batch inference pipeline
- Online feature validation
- Secret management with Kubernetes Secrets or external vaults
- Cloud object storage for artifacts
- Automated model retraining triggers

---

# References

- Wine Quality Dataset (UCI Machine Learning Repository)
- Scikit-learn
- MLflow
- Optuna
- FastAPI
- Docker
- Kubernetes
- Prometheus
- Grafana
- Apache Airflow
- GitHub Actions
- DVC

---

# Author

**Sheejith KV**

Senior Site Reliability Engineer

This project demonstrates a production-oriented MLOps workflow covering the complete machine learning lifecycle from data ingestion to deployment, monitoring, orchestration, and automated testing.

---

# Submission Checklist

Before submission, verify:

- [x] Project builds successfully
- [x] Python modules compile
- [x] Data preprocessing works
- [x] Model training works
- [x] Optuna tuning works
- [x] MLflow experiment tracking works
- [x] Model registry works
- [x] FastAPI endpoints respond correctly
- [x] Docker image builds successfully
- [x] Kubernetes manifests validate
- [x] Prometheus metrics endpoint works
- [x] Grafana dashboard is included
- [x] Airflow DAG is present
- [x] GitHub Actions workflows are configured
- [x] Automated tests pass
- [x] Documentation is complete

---

# Conclusion

This project implements an end-to-end MLOps pipeline for wine quality prediction using modern open-source technologies. It demonstrates reproducible data processing, model training, experiment tracking, hyperparameter optimization, model versioning, API serving, containerization, orchestration, monitoring, workflow automation, and continuous integration. The project is structured to be maintainable, extensible, and suitable as a reference implementation for production-style machine learning workflows.