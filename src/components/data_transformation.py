from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.components.data_ingestion import DataIngestionConfig
import numpy as np

from src.exception import CustomException
from src.logger import logger

from dataclasses import dataclass
import os
import sys
import pickle

@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path:str = os.path.join("artifacts","preprocessor.pkl")

    train_processes_path:str = os.path.join("data","processed","train_processed.csv")

    test_processes_path:str = os.path.join("data","processed","test_processed.csv")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logger.info("Data transformation Started..")

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logger.info("Train and Test data loaded successfully...")

            target_column = "SalePrice"

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            ### seprating numerical and categorical columns.
            numerical_columns = input_feature_train_df.select_dtypes(
                exclude="object"
            ).columns

            categorical_columns = input_feature_train_df.select_dtypes(
                include=["object", "string"]
            ).columns

            logger.info(
            "Numerical and categorical columns identified successfully."
            )

            preprocessor = self.get_data_transformer_object(numerical_columns,categorical_columns)

            logger.info("Applying preprocessing object on training and testing data.")

            input_feature_train_arr = preprocessor.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessor.transform(
                input_feature_test_df
            )

            # Convert sparse matrix to dense NumPy array (if required)
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()

            logger.info("Preprocessing completed successfully.")

            # Debugging shapes
            logger.info(f"Train transformed shape: {input_feature_train_arr.shape}")
            logger.info(f"Train target shape: {target_feature_train_df.shape}")

            logger.info(f"Test transformed shape: {input_feature_test_arr.shape}")
            logger.info(f"Test target shape: {target_feature_test_df.shape}")

            logger.info("Saving preprocessing object.")

            with open(
                self.data_transformation_config.preprocessor_obj_file_path,
                "wb"
            ) as file:
                pickle.dump(preprocessor, file)

            logger.info("Preprocessor object saved successfully.")

            train_arr = np.c_[
                input_feature_train_arr,
                target_feature_train_df.to_numpy()
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                target_feature_test_df.to_numpy()
            ]

            logger.info(f"Final train array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")

            logger.info("Training and Testing arrays created successfully.")    

            os.makedirs(os.path.dirname(self.data_transformation_config.train_processes_path),exist_ok= True)

            pd.DataFrame(train_arr).to_csv(self.data_transformation_config.train_processes_path,index=False,header=True)

            pd.DataFrame(test_arr).to_csv(self.data_transformation_config.test_processes_path,index=False,header=True)

            logger.info("Train and Test processed file created successfully")
            print("Train and Test processed file created successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logger.error(CustomException(e,sys))
            raise CustomException(e,sys)

    def get_data_transformer_object(self,
        numerical_columns,
        categorical_columns):

        try:
            logger.info("Creating Data Transformation Pipeline.")

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]
            )

            logger.info("Numerical and Categorical Pipelines created successfully.")

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info("ColumnTransformer created successfully.")

            return preprocessor

        except Exception as e:

            logger.error(CustomException(e, sys))
            raise CustomException(e, sys)

if __name__=="__main__":

    data = DataTransformation()

    initiate_data = DataIngestionConfig()

    train_data = initiate_data.train_data_path
    test_data = initiate_data.test_data_path

    data.initiate_data_transformation(train_data,test_data)
