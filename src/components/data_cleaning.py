

import sys
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion


@dataclass
class DataCleaningConfig:
    drop_duplicates: bool = True
    drop_missing: bool = True


class DataCleaning:
    def __init__(self, config: DataCleaningConfig = DataCleaningConfig()):
        self.config = config

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Data cleaning started")
            logger.info(f"Initial dataset shape: {df.shape}")

            # Remove duplicates
            if self.config.drop_duplicates:
                before = df.shape[0]
                df = df.drop_duplicates()
                logger.info(f"Duplicates removed: {before - df.shape[0]}")

            # Handle missing values
            if self.config.drop_missing:
                missing = df.isnull().sum().sum()
                if missing > 0:
                    logger.info(f"Missing values found: {missing}")
                    df = df.dropna()
                    logger.info("Rows with missing values removed")

            logger.info(f"Dataset shape after cleaning: {df.shape}")
            logger.info("Data cleaning completed")

            return df

        except Exception as e:
            logger.error("Error during data cleaning")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        clean_df = cleaner.clean_data(df)

        print(clean_df.head())

    except Exception as e:
        raise CustomException(e, sys)