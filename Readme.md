# 🏠 Advanced House Price Prediction API

A production-ready end-to-end Machine Learning project that predicts house prices using the **Ames Housing Dataset**. The project follows a modular software architecture and exposes the trained model through a **FastAPI REST API**. The entire application is containerized using **Docker**, making it portable and deployment-ready.

---

# 🚀 Project Overview

The objective of this project is to build a scalable Machine Learning application capable of predicting house prices from various property features.

The project follows industry-standard software engineering practices, including:

* Modular project architecture
* Object-Oriented Programming (OOP)
* Data preprocessing pipelines
* Model training and evaluation
* Model serialization
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
            Model Evaluation
                     │
                     ▼
        model.pkl + preprocessor.pkl
                     │
──────────────────────────────────────────────────────

                  Inference Pipeline

            FastAPI Prediction API
                     │
                     ▼
             JSON Request Body
                     │
                     ▼
          Pydantic Validation
                     │
                     ▼
             Pandas DataFrame
                     │
                     ▼
      preprocessor.transform(df)
                     │
                     ▼
            model.predict()
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
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   └── data_validation_report.txt
│
├── data/
│   ├── raw/
│   └── processed/
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

* Reads raw dataset
* Creates artifact directory
* Splits data into train and test datasets
* Stores:

  * raw.csv
  * train.csv
  * test.csv

---

## 2. Data Validation

Performs validation before preprocessing.

Validation includes:

* Dataset availability
* Missing values
* Duplicate records
* Dataset shape

Validation report is generated inside:

```text
artifacts/data_validation_report.txt
```

---

## 3. Data Transformation

A Scikit-Learn preprocessing pipeline is created using **ColumnTransformer**.

### Numerical Pipeline

* Median Imputation
* StandardScaler

### Categorical Pipeline

* Most Frequent Imputation
* OneHotEncoder

The preprocessing pipeline is serialized as:

```text
artifacts/preprocessor.pkl
```

---

## 4. Model Training

Multiple regression algorithms are trained and compared.

Models are evaluated using:

* R² Score
* Mean Absolute Error
* Mean Squared Error
* Root Mean Squared Error

The best-performing model is selected and saved as:

```text
artifacts/model.pkl
```

---

# 🌐 FastAPI Prediction Service

The trained model is deployed as a REST API using **FastAPI**.

## Workflow

```text
Client
   │
   ▼
POST /predict
   │
   ▼
Pydantic Validation
   │
   ▼
Convert JSON → DataFrame
   │
   ▼
Preprocessor
   │
   ▼
Machine Learning Model
   │
   ▼
Predicted House Price
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

Build Docker Image

```bash
docker build -t house-price-api .
```

Run Docker Container

```bash
docker run -p 8000:8000 house-price-api
```

API will be available at

```text
http://localhost:8000
```

Swagger Documentation

```text
http://localhost:8000/docs
```

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Pydantic
* Uvicorn
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
* Model serialization
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

Windows

```bash
myenv\Scripts\activate
```

Linux/macOS

```bash
source myenv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run FastAPI

```bash
uvicorn app.main:app --reload
```

Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# 📈 Future Improvements

* MLflow integration
* Model versioning
* Unit testing
* CI/CD pipeline
* AWS EC2 deployment
* Kubernetes deployment
* Monitoring and observability

---

# 👨‍💻 Author

**Hitesh Yerekar**

Machine Learning Engineer

