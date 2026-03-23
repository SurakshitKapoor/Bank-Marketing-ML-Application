
import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import CustomException


@dataclass
class DataIngestionConfig:
    data_path: str = "data/dataset.csv"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info("Data ingestion started")

            if not os.path.exists(self.config.data_path):
                logger.error(f"Dataset not found: {self.config.data_path}")
                raise FileNotFoundError(f"{self.config.data_path} not found")

            logger.info(f"Reading dataset from {self.config.data_path}")

            df = pd.read_csv(self.config.data_path)

            logger.info(f"Dataset loaded successfully | Shape: {df.shape}")

            return df

        except Exception as e:
            logger.error("Error during data ingestion")
            raise CustomException(e, sys)
        
    

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    print(df.head())