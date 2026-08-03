from fastapi import FastAPI
from app.schema import HouseData
import pickle
import pandas as pd

from src.exception import CustomException
from src.logger import logger

import mlflow
import os
import sys

#pipeline = mlflow.pyfunc.load_model("artifacts/model")

with open("artifacts//model.pkl","rb") as f:
    pipeline = pickle.load(f)

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="House Price Prediction using Machine Learning"
)

@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running..."
    }

@app.post("/predict")
def predict(data: HouseData):
    try:
        logger.info("Loading the input data")

        df = pd.DataFrame([data.model_dump(by_alias=True)])

        logger.info("User data successfully loaded")

        prediction = pipeline.predict(df)

        logger.info("Prediction done successfully")

        return {
            "status": "success",
            "prediction": float(prediction[0])
        }
    except Exception as e:

        logger.error(str(CustomException(e,sys)))
        raise CustomException(e,sys)

        

