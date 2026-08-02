# 🏠 Advanced House Price Prediction API

A production-ready end-to-end Machine Learning project that predicts house prices using the **Ames Housing Dataset**. The project follows an industry-standard modular architecture, integrates **MLflow Experiment Tracking** and **Model Registry** for model lifecycle management, exposes predictions through a **FastAPI REST API**, and is fully containerized using **Docker**.

---

# 🚀 Project Overview

The objective of this project is to build a scalable Machine Learning application capable of predicting house prices from various property features while following production-grade MLOps practices.

The project includes:

* Modular project architecture
* Object-Oriented Programming (OOP)
* Data preprocessing pipelines
* Hyperparameter tuning
* MLflow Experiment Tracking
* MLflow Model Registry
* Production model management
* FastAPI REST API
* Pydantic request validation
* Docker containerization
* Logging
* Custom exception handling

---

# 🏗 Project Architecture

```text
                           Training Pipeline

                     Raw Dataset
                          │
                          ▼
                   Data Ingestion
                          │
                          ▼
                  Data Validation
                          │
                          ▼
               Data Transformation
                          │
                          ▼
                  Model Training
                          │
                          ▼
             Hyperparameter Tuning
                          │
                          ▼
             MLflow Experiment Tracking
                          │
                          ▼
              MLflow Model Registry
                          │
                          ▼
                 Production Pipeline

────────────────────────────────────────────────────────────────────────

                          Inference Pipeline

                  FastAPI Prediction API
                          │
                          ▼
                    JSON Request
                          │
                          ▼
                 Pydantic Validation
                          │
                          ▼
                 Pandas DataFrame
                          │
                          ▼
        Load Production Pipeline from MLflow Registry
                          │
                          ▼
                  Pipeline.predict()
                          │
                          ▼
                   JSON Response
```

---

# 📂 Project Structure

```text
Advanced_House_Price_Prediction/
│
├── app/
│   ├── main.py
│   └── schema.py
│
├── artifacts/
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── data_validation_report.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── mlruns/
│
├── notebooks/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

---

# ⚙️ Machine Learning Pipeline

## 1. Data Ingestion

Responsibilities:

* Reads the raw dataset
* Creates the artifacts directory
* Splits data into train and test datasets

Generated artifacts:

* raw.csv
* train.csv
* test.csv

---

## 2. Data Validation

Validation checks include:

* Dataset availability
* Missing values
* Duplicate records
* Dataset shape

Validation report:

```text
artifacts/data_validation_report.txt
```

---

## 3. Data Transformation

A preprocessing pipeline is created using **Scikit-Learn ColumnTransformer**.

### Numerical Pipeline

* Median Imputation
* StandardScaler

### Categorical Pipeline

* Most Frequent Imputation
* OneHotEncoder

The preprocessing object is later combined with the trained model into a single Scikit-Learn **Pipeline**.

---

## 4. Model Training

Multiple regression algorithms are trained:

* Linear Regression
* Ridge
* Lasso
* Random Forest
* Gradient Boosting
* XGBoost

The models are initially evaluated using:

* R² Score

The top-performing models are then hyperparameter tuned using **GridSearchCV**.

Final evaluation metrics:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

# 📊 MLflow Integration

The project integrates **MLflow** for experiment tracking and model lifecycle management.

## Experiment Tracking

Each experiment automatically logs:

* Model Name
* Hyperparameters
* Evaluation Metrics
* Trained Pipeline Artifact

Tracked Metrics:

* R² Score
* MAE
* RMSE

---

## Model Registry

After selecting the best-performing model:

* The complete Scikit-Learn Pipeline (Preprocessor + Model) is registered in the **MLflow Model Registry**.
* The production version is promoted through the registry.
* FastAPI always loads the current Production model.

This eliminates the need to manually manage separate preprocessing and model files during inference.

---

# 🌐 FastAPI Prediction Service

The production pipeline is loaded directly from the **MLflow Model Registry**.

Workflow:

```text
Client
   │
POST /predict
   │
Pydantic Validation
   │
Convert JSON → DataFrame
   │
Load Production Pipeline
from MLflow Model Registry
   │
Pipeline.predict()
   │
Prediction Response
```

---

# 📌 API Endpoints

## Home Endpoint

```http
GET /
```

Response

```json
{
  "message": "House Price Prediction API is running..."
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Sample Response

```json
{
  "status": "success",
  "prediction": 212175.29
}
```

---

# 🐳 Docker Support

The application is fully containerized.

## Build Docker Image

```bash
docker build -t house-price-api .
```

## Run Docker Container

```bash
docker run -p 8000:8000 house-price-api
```

Application:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Scikit-Learn Pipeline
* GridSearchCV
* XGBoost
* FastAPI
* Pydantic
* Uvicorn
* MLflow
* Docker
* Pickle
* Logging
* Dataclasses
* Object-Oriented Programming

---

# ✨ Features

* Production-style project architecture
* Modular ML pipeline
* Automated preprocessing pipeline
* Hyperparameter Optimization
* MLflow Experiment Tracking
* MLflow Model Registry
* Production model loading
* FastAPI REST API
* Pydantic request validation
* Docker containerization
* Interactive Swagger UI
* Custom exception handling
* Custom logging
* Reusable codebase

---

# ▶️ Getting Started

## Clone Repository

```bash
git clone <repository-url>
```

## Create Virtual Environment

```bash
python -m venv myenv
```

## Activate Environment

### Windows

```bash
myenv\Scripts\activate
```

### Linux/macOS

```bash
source myenv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start MLflow Server

```bash
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--host 0.0.0.0 \
--port 5000
```

## Run FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 📈 Future Improvements

* GitHub Actions CI/CD
* Docker Compose
* AWS Deployment (EC2, ECR, ECS/App Runner)
* Kubernetes Deployment
* Model Monitoring
* Data Version Control (DVC)
* Monitoring & Observability (Prometheus/Grafana)

---

# 👨‍💻 Author

**Hitesh Yerekar**

Machine Learning Engineer
