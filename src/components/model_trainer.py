import pandas as pd
import numpy as np

import os
import sys
import pickle
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import GridSearchCV

import mlflow 
import mlflow.sklearn

import mlflow.xgboost

with open("artifacts//preprocessor.pkl","rb") as file:

    preprocessor = pickle.load(file)

@dataclass
class ModelTrainerConfig:

    train_model_path:str = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):

        self.model_config = ModelTrainerConfig()

    def initiate_model_training(self,train_data_path,test_data_path):

        try:
            logger.info("Model training processed start")

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logger.info("Processed train and test data loaded successfully.")

            # Training data.
            X_train = train_data.iloc[:,:-1].values
            y_train = train_data.iloc[:,-1].values

            # Testing data.
            X_test = test_data.iloc[:,:-1].values
            y_test = test_data.iloc[:,-1].values

            logger.info("Features and target separated successfully.")

            models = {
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'RandomForest':RandomForestRegressor(),
                'GradientBoosting':GradientBoostingRegressor(),
                'XGBoost':XGBRegressor()
            }

            params = {

                        "LinearRegression": {
                            "fit_intercept": [True, False]
                        },

                        "Ridge": {
                            "alpha": [0.1, 1, 10]
                        },

                        "Lasso": {
                            "alpha": [0.001, 0.01, 0.1]
                        },

                        "RandomForest": {
                            "n_estimators": [100, 200],
                            "max_depth": [10, 20, None]
                        },

                        "GradientBoosting": {
                            "n_estimators": [100, 200],
                            "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 5]
                        },

                        "XGBoost": {
                            "n_estimators": [100, 200],
                            "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 5]
                        }

                    }
            mlflow.set_experiment("House Price Prediction")

            model_score ={}

            for model_name, model in models.items():

                logger.info(f"Training {model_name}...")

                model.fit(X_train,y_train)

                prediction = model.predict(X_test)

                score = r2_score(y_test,prediction)

                logger.info(f"{model_name} R2 Score : {score:.4f}")
                
                model_score[model_name] = score

            sorted_models = sorted(
                                    model_score.items(),
                                     key=lambda x: x[1],
                                    reverse=True
                                )
            
            top_3_models = sorted_models[:3]
            logger.info(f"Top 3 Models: {top_3_models}")

            tuned_model_score = {}
            best_models = {}

            for model_name, score in top_3_models:

                with mlflow.start_run(run_name=model_name):

                    logger.info(f"Tuning {model_name}")

                    model = models[model_name]

                    param_grid = params[model_name]

                    grid_search = GridSearchCV(estimator= model,param_grid=param_grid,cv=5,scoring="r2", n_jobs=-1)

                    grid_search.fit(X_train,y_train)

                    mlflow.log_param("model_name", model_name)

                    for param_name, param_value in grid_search.best_params_.items():
                        mlflow.log_param(param_name, param_value)

                    tuned_model = grid_search.best_estimator_

                    prediction = tuned_model.predict(X_test)

                    r2 = r2_score(y_test, prediction)
                    mae = mean_absolute_error(y_test, prediction)
                    rmse = root_mean_squared_error(y_test, prediction)

                    mlflow.log_metric("R2 Score", r2)
                    mlflow.log_metric("MAE", mae)
                    mlflow.log_metric("RMSE", rmse)

                    pipeline = Pipeline([
                                            ("preprocessor", preprocessor),
                                            ("model", tuned_model)
                                        ])
                    
                    mlflow.sklearn.log_model(
                            sk_model=pipeline,
                            artifact_path="model"
                        )

                    tuned_model_score[model_name] = r2
                    best_models[model_name] = pipeline

                    logger.info(
                                f"{model_name} "
                                f"Best Params: {grid_search.best_params_} "
                                f"R2 Score: {r2:.4f}"
                            )
            best_model_name = max(tuned_model_score,key=tuned_model_score.get)

            best_model_score = tuned_model_score[best_model_name]

            best_model = best_models[best_model_name]

            os.makedirs(
                        os.path.dirname(self.model_config.train_model_path),
                        exist_ok=True
                        )
                
            with open(self.model_config.train_model_path,'wb') as f:
                pickle.dump(best_model,f)

            logger.info(
                f"Best Model : {best_model_name} | "
                f"R2 Score : {best_model_score:.4f}"
            )

            logger.info("Saving best model...")

            return (
                    best_model,
                    best_model_name,
                    best_model_score,
                    self.model_config.train_model_path
                )

        except Exception as e:

            logger.error(CustomException(e,sys))
            raise CustomException(e,sys)


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        train_data_path="data/processed/train_processed.csv",
        test_data_path="data/processed/test_processed.csv"
    )
