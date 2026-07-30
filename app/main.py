from fastapi import FastAPI
from app.schema import HouseData
import pickle
import pandas as pd

from src.exception import CustomException
from src.logger import logger

import os
import sys

def load_preprocessor():
    with open("artifacts//preprocessor.pkl",'rb') as file:
        preprocessor_obj = pickle.load(file)
    return preprocessor_obj

def load_model():
    with open("artifacts//model.pkl","rb") as file:
        model_obj = pickle.load(file)
    return model_obj

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="House Price Prediction using Machine Learning"
)

preprocessor = load_preprocessor()
model = load_model()


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

        preprocess_data = preprocessor.transform(df)

        logger.info("Successfully preprocessed data")

        prediction = model.predict(preprocess_data)

        logger.info("Prediction done successfully")

        return {
            "status": "success",
            "prediction": float(prediction[0])
        }
    except Exception as e:

        logger.error(str(CustomException(e,sys)))
        raise CustomException(e,sys)

        

