import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestionConfig

from dataclasses import dataclass


@dataclass
class DataValidationConfig:
    data_validation_path: str = os.path.join("artifacts", "data_validation_report.txt")


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def initiate_data_validation(self, train_data, test_data):
        try:
            logger.info("Data Validation Started")

            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logger.info("Train and Test data loaded successfully.")

            validation_status = True

            report = []

            if train_df.empty:
                validation_status = False
                report.append("Train dataset is empty")

            if test_df.empty:
                validation_status = False
                report.append("Test dataset is empty")

            # Checking dataset shape
            report.append(f"Train shape:{train_df.shape}")
            report.append(f"Test shape:{test_df.shape}")

            # checking missing values in dataset.
            train_missing_data = train_df.isnull().sum()
            test_missing_data = test_df.isnull().sum()

            report.append(f"Train missing values:{train_missing_data}")
            report.append(f"Test missing values:{test_missing_data}")

            # checking the duplicates data.
            train_duplicated_data = train_df.duplicated().sum()
            test_duplicated_data = test_df.duplicated().sum()

            report.append(f"Train duplicated data:{train_duplicated_data}")
            report.append(f"Test duplicated data:{test_duplicated_data}")

            if validation_status:
                report.append("Data validation passed")
                logger.info("Data validation successfully")

            else:
                report.append("Data validation failed")
                logger.info("Data validation failed")

            # save validation report.

            with open(self.validation_config.data_validation_path, "w") as file:
                for line in report:
                    file.write(line + "\n")
            logger.info("Validation report generated successfully.")

            return validation_status

        except Exception as e:
            logger.error(CustomException(e, sys))

            raise CustomException(e, sys)


if __name__ == "__main__":
    data = DataValidation()
    ingestion_config = DataIngestionConfig()

    train_data = ingestion_config.train_data_path
    test_data = ingestion_config.test_data_path

    data.initiate_data_validation(train_data, test_data)
