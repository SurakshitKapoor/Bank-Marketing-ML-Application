

import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.feature_engineering import FeatureEngineering


@dataclass
class DataPreprocessingConfig:
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "conversion_status"


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig = DataPreprocessingConfig()):
        self.config = config

    def preprocess_data(self, df: pd.DataFrame):

        try:
            logger.info("Data preprocessing started")

            # -----------------------------
            # Separate input and target
            # -----------------------------
            X = df.drop(columns=self.config.target_column)
            y = df[self.config.target_column]

            logger.info("Separated features and target")

            # -----------------------------
            # Train test split
            # -----------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            logger.info(f"Train shape: {X_train.shape}")
            logger.info(f"Test shape: {X_test.shape}")

            # -----------------------------
            # Identify column types
            # -----------------------------
            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns

            numerical_cols = X_train.select_dtypes(
                include="number"
            ).columns

            logger.info(f"Categorical columns: {list(categorical_cols)}")
            logger.info(f"Numerical columns: {list(numerical_cols)}")

            # -----------------------------
            # One Hot Encoding
            # -----------------------------
            ohe = OneHotEncoder(drop="first", handle_unknown="ignore")

            X_train_enc = ohe.fit_transform(X_train[categorical_cols])
            X_test_enc = ohe.transform(X_test[categorical_cols])

            logger.info("Categorical features encoded")

            # -----------------------------
            # Label Encoding target
            # -----------------------------
            le = LabelEncoder()

            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            logger.info("Target variable encoded")

            # -----------------------------
            # Outlier Handling (Winsorization)
            # -----------------------------
            for col in numerical_cols:

                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col] = X_test[col].clip(lower, upper)

            logger.info("Outliers handled using winsorization")

            # -----------------------------
            # Feature Scaling
            # -----------------------------
            scaler = StandardScaler()

            X_train_sc = scaler.fit_transform(X_train[numerical_cols])
            X_test_sc = scaler.transform(X_test[numerical_cols])

            logger.info("Numerical features scaled")

            # -----------------------------
            # Combine features
            # -----------------------------
            # -----------------------------

            X_train_final = np.hstack((X_train_sc, X_train_enc.toarray()))
            X_test_final = np.hstack((X_test_sc, X_test_enc.toarray()))

            # -----------------------------
            # Create feature names
            # -----------------------------
            ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

            final_feature_names = list(numerical_cols) + list(ohe_feature_names)

            logger.info(f"Total final features: {len(final_feature_names)}")
            logger.info(f"Feature names: {final_feature_names}")

            # Convert to DataFrame for readability
            X_train_final = pd.DataFrame(X_train_final, columns=final_feature_names)
            X_test_final = pd.DataFrame(X_test_final, columns=final_feature_names)

            logger.info(f"Final training shape: {X_train_final.shape}")
            logger.info(f"Final testing shape: {X_test_final.shape}")

            logger.info("Data preprocessing completed successfully")

            return (
                X_train_final,
                X_test_final,
                y_train,
                y_test,
                ohe,
                scaler,
                le,
                numerical_cols,
                categorical_cols
            )

        except Exception as e:
            logger.error("Error occurred during data preprocessing")
            raise CustomException(e, sys)


# -----------------------------
# Test Pipeline
# -----------------------------
if __name__ == "__main__":

    try:
        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        clean_df = cleaner.clean_data(df)

        fe = FeatureEngineering()
        feature_df = fe.engineer_features(clean_df)

        preprocessor = DataPreprocessing()
        X_train, X_test, y_train, y_test, *_ = preprocessor.preprocess_data(feature_df)

        print("Train Shape:", X_train.shape)
        print("Test Shape:", X_test.shape)

    except Exception as e:
        raise CustomException(e, sys)