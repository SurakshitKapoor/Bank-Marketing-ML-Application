
import sys
import os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion


@dataclass
class DataCleaningConfig:
    processed_data_dir: str = os.path.join("data", "processed")

    X_train_path: str = os.path.join("data", "processed", "X_train.csv")
    X_test_path: str = os.path.join("data", "processed", "X_test.csv")
    y_train_path: str = os.path.join("data", "processed", "y_train.csv")
    y_test_path: str = os.path.join("data", "processed", "y_test.csv")

    test_size: float = 0.20
    random_state: int = 20


class DataCleaning:

    def __init__(self, config: DataCleaningConfig = DataCleaningConfig()):
        self.config = config

    def clean_and_split(self, df: pd.DataFrame):

        try:
            logger.info("Data cleaning started")
            logger.info(f"Initial dataset shape: {df.shape}")

            # Remove duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                df = df.drop_duplicates()
                logger.info(f"Duplicates removed: {duplicates}")

            # Check missing values
            missing = df.isnull().sum().sum()
            logger.info(f"Total missing values: {missing}")

            if missing > 0:
                df = df.dropna()
                logger.info("Rows with missing values removed")

            logger.info(f"Dataset shape after cleaning: {df.shape}")

            # -------- SPLIT INPUT & TARGET -------- #

            logger.info("Separating input and target variables")

            X = df.drop(columns=["conversion_status"])
            y = df[["conversion_status"]]

            logger.info(f"X shape: {X.shape}")
            logger.info(f"y shape: {y.shape}")

            # -------- TRAIN TEST SPLIT -------- #

            logger.info("Train-test split started")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")

            # -------- SAVE DATASETS -------- #

            os.makedirs(self.config.processed_data_dir, exist_ok=True)

            X_train.to_csv(self.config.X_train_path, index=False)
            X_test.to_csv(self.config.X_test_path, index=False)
            y_train.to_csv(self.config.y_train_path, index=False)
            y_test.to_csv(self.config.y_test_path, index=False)

            logger.info("Train-test datasets saved successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Error during data cleaning and splitting")
            raise CustomException(e, sys)


if __name__ == "__main__":

    try:

        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        X_train, X_test, y_train, y_test = cleaner.clean_and_split(df)

        print("Data cleaning and split completed")

    except Exception as e:
        raise CustomException(e, sys)