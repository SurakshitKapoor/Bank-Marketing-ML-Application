

import sys
import os
import pandas as pd
import numpy as np

from src.utils.exception import CustomException
from src.utils.file_ops import load_object
from src.utils.logger import logger


class PredictPipeline:

    def __init__(self):

        try:
            artifacts_dir = "artifacts"

            self.model = load_object(os.path.join(artifacts_dir, "best_model.pkl"))
            self.encoder = load_object(os.path.join(artifacts_dir, "encoder.pkl"))
            self.scaler = load_object(os.path.join(artifacts_dir, "scaler.pkl"))
            self.label_encoder = load_object(os.path.join(artifacts_dir, "label_encoder.pkl"))

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------
    # Feature Engineering (same logic as training)
    # -------------------------------------------------------
    def feature_engineering(self, df):

        try:

            df = df.copy()

            # Age group
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 30, 50, 100],
                labels=["young", "mid_age", "senior"]
            )

            # Binary features
            df["is_previous_success"] = (
                df["previous_campaign_outcome"] == "successful"
            ).astype(int)

            df["contacted_via_mobile"] = (
                df["communication_channel"] == "mobile"
            ).astype(int)

            df["is_month_start"] = (df["call_day"] <= 5).astype(int)
            df["is_month_end"] = (df["call_day"] >= 25).astype(int)

            # Buckets
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

            # Season mapping
            season_map = {
                "December": "winter", "January": "winter", "February": "winter",
                "March": "spring", "April": "spring", "May": "spring",
                "June": "summer", "July": "summer", "August": "summer",
                "September": "fall", "October": "fall", "November": "fall"
            }

            df["campaign_season"] = df["call_month"].map(season_map)

            # Interaction feature
            df["total_talk_time"] = df["call_duration"] * df["call_frequency"]

            # Relative features
            freq_med = df["call_frequency"].median()
            dur_med = df["call_duration"].median()

            df["is_high_call_frequency"] = (df["call_frequency"] > freq_med).astype(int)

            df["contact_efficiency"] = np.where(
                (df["call_duration"] > dur_med) &
                (df["call_frequency"] <= freq_med),
                1,
                0
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------
    def predict(self, features: pd.DataFrame):

        try:

            df = self.feature_engineering(features)

            # SAME columns used during training
            categorical_cols = [
                "occupation",
                "education_level",
                "marital_status",
                "communication_channel",
                "call_month",
                "previous_campaign_outcome",
                "age_group",
                "call_duration_bucket",
                "day_of_month_phase",
                "campaign_season"
            ]
            #  ['age', 'call_day', 'call_duration', 'call_frequency', 'is_previous_success', 'contacted_via_mobile', 'is_month_start', 'is_month_end', 'total_talk_time', 'is_high_call_frequency', 'contact_efficiency']
            numerical_cols = [
                "age",
                "call_day",
                "call_duration",
                "call_frequency",
                "is_previous_success",
                "contacted_via_mobile",
                "is_month_start",
                "is_month_end",
                "total_talk_time",
                "is_high_call_frequency",
                "contact_efficiency"
            ]

            # Encode categorical
            X_enc = self.encoder.transform(df[categorical_cols])

            X_enc_df = pd.DataFrame(
                X_enc,
                columns=self.encoder.get_feature_names_out(categorical_cols)
            )

            # Scale numerical
            X_sc = self.scaler.transform(df[numerical_cols])

            X_sc_df = pd.DataFrame(
                X_sc,
                columns=numerical_cols
            )

            # Combine
            X_final = pd.concat(
                [X_sc_df.reset_index(drop=True),
                 X_enc_df.reset_index(drop=True)],
                axis=1
            )

            # Prediction
            preds = self.model.predict(X_final)

            preds = self.label_encoder.inverse_transform(preds)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------
# Custom Input Class
# -------------------------------------------------------

class CustomData:

    def __init__(
        self,
        age,
        occupation,
        marital_status,
        education_level,
        communication_channel,
        call_day,
        call_month,
        call_duration,
        call_frequency,
        previous_campaign_outcome
    ):

        self.age = age
        self.occupation = occupation
        self.marital_status = marital_status
        self.education_level = education_level
        self.communication_channel = communication_channel
        self.call_day = call_day
        self.call_month = call_month
        self.call_duration = call_duration
        self.call_frequency = call_frequency
        self.previous_campaign_outcome = previous_campaign_outcome

    def get_data_as_dataframe(self):

        try:

            data_dict = {
                "age": [self.age],
                "occupation": [self.occupation],
                "marital_status": [self.marital_status],
                "education_level": [self.education_level],

                "communication_channel": [self.communication_channel],
                "call_day": [self.call_day],
                "call_month": [self.call_month],
                "call_duration": [self.call_duration],
                "call_frequency": [self.call_frequency],
                "previous_campaign_outcome": [self.previous_campaign_outcome]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------
# Test Prediction Pipeline
# -------------------------------------------------------

if __name__ == "__main__":

    try:

        logger.info("Testing prediction pipeline")

        data = CustomData(
            age=35,
            occupation="management",
            marital_status="married",
            education_level="tertiary",
            
            communication_channel="mobile",
            call_day=5,
            call_month="May",
            call_duration=180,
            call_frequency=2,
            previous_campaign_outcome="unknown"
        )

        df = data.get_data_as_dataframe()

        logger.info(f"Input data:\n{df}")

        pipeline = PredictPipeline()

        prediction = pipeline.predict(df)

        logger.info(f"Prediction: {prediction}")

        print("Prediction:", prediction)

    except Exception as e:
        raise CustomException(e, sys)