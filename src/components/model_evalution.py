from dataclasses import dataclass
import os
import sys
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.exception import CustomException
from src.logger import logger


@dataclass
class ModelEvaluationConfig:

    accepted_model_path:str = os.path.join("artifacts","model.pkl")

class ModelEvaluation:

    def __init__(self):

        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self,processed_test_path):
        try:
            logger.info("Loding the processed test file")

            test_df = pd.read_csv(processed_test_path)

            X_test = test_df.iloc[:,:-1].values
            y_test = test_df.iloc[:,-1].values

            with open(self.model_evaluation_config.accepted_model_path,'rb')as file:

                logger.info("Loading the model.")
                model = pickle.load(file)

            prediction = model.predict(X_test)

            # Metrics
            r2 = r2_score(y_test, prediction)
            mae = mean_absolute_error(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)
            rmse = np.sqrt(mse)

            logger.info(
                f"R2 Score  : {r2:.4f}\n"
                f"MAE Score : {mae:.4f}\n"
                f"MSE Score : {mse:.4f}\n"
                f"RMSE Score: {rmse:.4f}"
            )
            return {
                    'r2_score':r2,
                    'mae':mae,
                    'mse':mse,
                    'rmse':rmse  
                    }

        except Exception as e:

            logger.error(CustomException(e,sys))

            raise CustomException(e,sys)

        
        




            
       

