
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning


@dataclass
class FeatureEngineeringConfig:
    create_age_group: bool = True
    create_call_duration_bucket: bool = True


class FeatureEngineering:

    def __init__(self, config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.config = config

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            logger.info("Feature engineering started")

            # -----------------------------
            # Age Group Feature
            # -----------------------------
            if self.config.create_age_group:
                df["age_group"] = pd.cut(
                    df["age"],
                    bins=[0, 30, 50, 100],
                    labels=["young", "mid_age", "senior"]
                ).astype("category")

                logger.info("Feature created: age_group")

            # -----------------------------
            # Previous Campaign Success
            # -----------------------------
            df["is_previous_success"] = (
                df["previous_campaign_outcome"] == "successful"
            ).astype("category")

            logger.info("Feature created: is_previous_success")

            # -----------------------------
            # Call Duration Bucket
            # -----------------------------
            if self.config.create_call_duration_bucket:
                df["call_duration_bucket"] = pd.cut(
                    df["call_duration"],
                    bins=[-1, 60, 300, np.inf],
                    labels=["short", "medium", "long"]
                ).astype("category")

                logger.info("Feature created: call_duration_bucket")

            # -----------------------------
            # High Call Frequency
            # -----------------------------
            df["is_high_call_frequency"] = (
                df["call_frequency"] > df["call_frequency"].median()
            ).astype("category")

            logger.info("Feature created: is_high_call_frequency")

            # -----------------------------
            # Contacted via Mobile
            # -----------------------------
            df["contacted_via_mobile"] = (
                df["communication_channel"] == "mobile"
            ).astype("category")

            logger.info("Feature created: contacted_via_mobile")

            # -----------------------------
            # Weekend Call
            # -----------------------------
            df["is_weekend_call"] = df["call_day"].apply(
                lambda x: 1 if x in [6, 7] else 0
            ).astype("category")

            logger.info("Feature created: is_weekend_call")

            # -----------------------------
            # Call Intensity (numeric)
            # -----------------------------
            df["call_intensity"] = df["call_duration"] / df["call_frequency"]

            logger.info("Feature created: call_intensity")

            # -----------------------------
            # Contact Efficiency
            # -----------------------------
            df["contact_efficiency"] = pd.Series(
                np.where(
                    (df["call_duration"] > df["call_duration"].median()) &
                    (df["call_frequency"] <= df["call_frequency"].median()),
                    1, 0
                ),
                index=df.index
            ).astype("category")

            logger.info("Feature created: contact_efficiency")

            logger.info(f"Feature engineering completed | Shape: {df.shape}")

            return df

        except Exception as e:
            logger.error("Error during feature engineering")
            raise CustomException(e, sys)


# ----------------------------------
# Test Pipeline
# ----------------------------------

if __name__ == "__main__":

    try:
        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        clean_df = cleaner.clean_data(df)

        fe = FeatureEngineering()
        feature_df = fe.engineer_features(clean_df)

        print(feature_df.head())

    except Exception as e:
        raise CustomException(e, sys)