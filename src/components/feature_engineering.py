

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exception import CustomException


@dataclass
class FeatureEngineeringConfig:
    featured_data_dir: str = os.path.join("data", "featured")


class FeatureEngineering:

    def __init__(self, config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.config = config

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            df = df.copy()

            # -----------------------------
            # Age Binning
            # -----------------------------
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 30, 50, 100],
                labels=["young", "mid_age", "senior"]
            )

            # -----------------------------
            # Binary Flags
            # -----------------------------
            df["is_previous_success"] = (
                df["previous_campaign_outcome"] == "successful"
            ).astype(int)

            df["contacted_via_mobile"] = (
                df["communication_channel"] == "mobile"
            ).astype(int)

            df["is_month_start"] = (df["call_day"] <= 5).astype(int)
            df["is_month_end"] = (df["call_day"] >= 25).astype(int)

            # -----------------------------
            # Buckets
            # -----------------------------
            df["call_duration_bucket"] = pd.cut(
                df["call_duration"],
                bins=[-1, 60, 300, np.inf],
                labels=["short", "medium", "long"]
            )

            df["day_of_month_phase"] = pd.cut(
                df["call_day"],
                bins=[0, 10, 20, 31],
                labels=["Early", "Mid", "Late"]
            )

            # -----------------------------
            # Seasonal Feature
            # -----------------------------
            season_map = {
                "December": "winter", "January": "winter", "February": "winter",
                "March": "spring", "April": "spring", "May": "spring",
                "June": "summer", "July": "summer", "August": "summer",
                "September": "fall", "October": "fall", "November": "fall"
            }

            df["campaign_season"] = df["call_month"].map(season_map)

            # -----------------------------
            # Interaction Features
            # -----------------------------
            df["total_talk_time"] = df["call_duration"] * df["call_frequency"]

            # -----------------------------
            # Relative Features
            # -----------------------------
            freq_med = df["call_frequency"].median()
            dur_med = df["call_duration"].median()

            df["is_high_call_frequency"] = (
                df["call_frequency"] > freq_med
            ).astype(int)

            df["contact_efficiency"] = np.where(
                (df["call_duration"] > dur_med) &
                (df["call_frequency"] <= freq_med),
                1,
                0
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def engineer_features(self, X_train, X_test, y_train, y_test):

        try:

            logger.info("Feature engineering started")

            X_train_fe = self._apply_feature_engineering(X_train)
            X_test_fe = self._apply_feature_engineering(X_test)

            logger.info("Feature engineering applied to train and test")

            # -----------------------------
            # Save datasets (optional but good for debugging)
            # -----------------------------
            os.makedirs(self.config.featured_data_dir, exist_ok=True)

            X_train_fe.to_csv(
                os.path.join(self.config.featured_data_dir, "X_train_fe.csv"),
                index=False
            )

            X_test_fe.to_csv(
                os.path.join(self.config.featured_data_dir, "X_test_fe.csv"),
                index=False
            )

            logger.info("Featured datasets saved")

            return X_train_fe, X_test_fe, y_train, y_test

        except Exception as e:
            logger.error("Feature engineering failed")
            raise CustomException(e, sys)


# -----------------------------------
# Test Pipeline
# -----------------------------------
if __name__ == "__main__":

    try:

        from src.components.data_ingestion import DataIngestion
        from src.components.data_cleaning import DataCleaning

        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        X_train, X_test, y_train, y_test = cleaner.clean_and_split(df)

        fe = FeatureEngineering()

        X_train_fe, X_test_fe, y_train, y_test = fe.engineer_features(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("Feature Engineering Completed")
        print("Train Shape:", X_train_fe.shape)
        print("Test Shape:", X_test_fe.shape)

    except Exception as e:
        raise CustomException(e, sys)