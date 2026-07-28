# 🏠 Advanced House Price Prediction

A production-style end-to-end Machine Learning project that predicts house prices using the Ames Housing dataset. This project follows a modular software architecture with separate components for data ingestion, validation, transformation, model training, evaluation, and prediction.

---

## 📌 Project Overview

The objective of this project is to build a robust Machine Learning pipeline capable of predicting house prices based on various property features.

The project is designed using industry-standard software engineering practices, including:

* Modular project structure
* Object-Oriented Programming (OOP)
* Custom exception handling
* Logging
* Configuration management using dataclasses
* Data preprocessing pipelines
* Model serialization
* Reusable components

---

## 📂 Project Structure

```text
Advanced_House_Price_Prediction/
│
├── artifacts/
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── data_validation_report.txt
│
├── data/
│   └── raw/
│       └── train.csv
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
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

# 🚀 Machine Learning Pipeline

## 1. Data Ingestion

* Loads the raw dataset
* Creates the artifacts directory
* Saves the raw dataset
* Splits the dataset into training and testing sets
* Stores:

  * train.csv
  * test.csv
  * raw.csv

---

## 2. Data Validation

Validates the dataset before preprocessing.

Validation includes:

* Empty dataset check
* Dataset shape
* Missing values
* Duplicate records

A validation report is generated inside the artifacts folder.

---

## 3. Data Transformation

Builds a preprocessing pipeline using Scikit-Learn.

### Numerical Features

* Median Imputation
* Standard Scaling

### Categorical Features

* Most Frequent Imputation
* One Hot Encoding

The project uses **ColumnTransformer** to combine both preprocessing pipelines.

The trained preprocessing object is saved as:

```text
artifacts/preprocessor.pkl
```

---

## 4. Model Training

* Trains multiple regression models
* Compares model performance
* Selects the best-performing model
* Saves the trained model

---

## 5. Model Evaluation

Evaluates model performance using regression metrics such as:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## 6. Prediction Pipeline

Loads:

* Trained Model
* Preprocessor

Transforms new user input and predicts the house price.

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Pickle
* Dataclasses
* Logging
* Object-Oriented Programming

---

# 🛠 Features

* Production-ready project structure
* Modular pipeline architecture
* Custom logging
* Custom exception handling
* Data validation report generation
* Automated preprocessing pipeline
* Model serialization
* Reusable codebase

---

# ▶️ How to Run

### Clone Repository

```bash
git clone <repository-url>
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Environment

Windows

```bash
myenv\Scripts\activate
```

Linux / macOS

```bash
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python app.py
```

---

# 📈 Future Improvements

* Hyperparameter tuning
* Feature selection
* Model versioning
* MLflow integration
* Docker containerization
* CI/CD pipeline
* Cloud deployment (AWS/Azure/GCP)

---

# 👨‍💻 Author

**Hitesh Yerekar**

Machine Learning Engineer

Skills:

* Python
* Machine Learning
* Deep Learning
* NLP
* Transformers
* LangChain
* LangGraph
* LLMs
* RAG
* Generative AI
* Agentic AI
* FastAPI
* AWS
* MySQL
