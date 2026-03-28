

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.file_ops import save_object


@dataclass
class DataPreprocessingConfig:
    artifacts_dir: str = os.path.join("artifacts")
    final_data_dir: str = os.path.join("data", "final")
    skew_threshold: float = 1.0


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig = DataPreprocessingConfig()):
        self.config = config

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series):

        try:

            logger.info("Data preprocessing started")

            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            # -----------------------------
            # Detect column types
            # -----------------------------
            categorical_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            numerical_cols = X_train.select_dtypes(
                include=["number"]
            ).columns.tolist()

            logger.info(f"Categorical columns: {categorical_cols}")
            logger.info(f"Numerical columns: {numerical_cols}")

            
            # -----------------------------
            # Skewness handling
            # -----------------------------
            skew_cols = []

            for col in numerical_cols:
                if abs(X_train[col].skew()) > self.config.skew_threshold:
                    skew_cols.append(col)

            logger.info(f"Highly skewed columns: {skew_cols}")

            for col in skew_cols:
                X_train[col] = np.log1p(X_train[col])
                X_test[col] = np.log1p(X_test[col])

            logger.info("Log transformation applied")

            
            # -----------------------------
            # Outlier clipping
            # -----------------------------
            for col in numerical_cols:

                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col] = X_test[col].clip(lower, upper)

            logger.info("Outliers handled using IQR clipping")

            
            
            # -----------------------------
            # One Hot Encoding
            # -----------------------------
            ohe = OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )

            X_train_enc = ohe.fit_transform(X_train[categorical_cols])
            X_test_enc = ohe.transform(X_test[categorical_cols])

            ohe_cols = ohe.get_feature_names_out(categorical_cols)

            X_train_enc_df = pd.DataFrame(
                X_train_enc,
                columns=ohe_cols,
                index=X_train.index
            )

            X_test_enc_df = pd.DataFrame(
                X_test_enc,
                columns=ohe_cols,
                index=X_test.index
            )

            logger.info("Categorical features encoded")

            
            
            # -----------------------------
            # Scaling numerical features
            # -----------------------------
            scaler = StandardScaler()

            X_train_sc = scaler.fit_transform(X_train[numerical_cols])
            X_test_sc = scaler.transform(X_test[numerical_cols])

            X_train_sc_df = pd.DataFrame(
                X_train_sc,
                columns=numerical_cols,
                index=X_train.index
            )

            X_test_sc_df = pd.DataFrame(
                X_test_sc,
                columns=numerical_cols,
                index=X_test.index
            )

            logger.info("Numerical features scaled")

            
            
            # -----------------------------
            # Merge encoded + scaled
            # -----------------------------
            X_train_final = pd.concat(
                [X_train_sc_df, X_train_enc_df],
                axis=1
            )

            X_test_final = pd.concat(
                [X_test_sc_df, X_test_enc_df],
                axis=1
            )

            logger.info(f"Final training shape: {X_train_final.shape}")
            logger.info(f"Final testing shape: {X_test_final.shape}")

            
            
            # -----------------------------
            # Encode target
            # -----------------------------
            le = LabelEncoder()

            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            logger.info("Target encoded")
            

            
            
            # -----------------------------
            # Save final datasets
            # -----------------------------
            
            os.makedirs(self.config.final_data_dir, exist_ok=True)
            
            X_train_final.to_csv(
            os.path.join(self.config.final_data_dir, "X_train_final.csv"),
                index=False )
            
            X_test_final.to_csv(
            os.path.join(self.config.final_data_dir, "X_test_final.csv"),
                index=False )
            
            pd.DataFrame(y_train, columns=["target"]).to_csv(
                os.path.join(self.config.final_data_dir, "y_train.csv"),
                index=False )
            
            pd.DataFrame(y_test, columns=["target"]).to_csv(
                os.path.join(self.config.final_data_dir, "y_test.csv"),
                index=False )
            
            
            
            
            # -----------------------------
            # Save artifacts
            # -----------------------------
            save_object(
                ohe,
                os.path.join(self.config.artifacts_dir, "encoder.pkl")
            )

            save_object(
                scaler,
                os.path.join(self.config.artifacts_dir, "scaler.pkl")
                
            )

            save_object(
                le,
                os.path.join(self.config.artifacts_dir, "label_encoder.pkl")
                
            )

            logger.info("Artifacts saved successfully")

            return X_train_final, X_test_final, y_train, y_test

        except Exception as e:
            logger.error("Error during preprocessing")
            raise CustomException(e, sys)



# ---------------------------------------
# Pipeline Test
# ---------------------------------------
if __name__ == "__main__":

    try:

        from src.components.data_ingestion import DataIngestion
        from src.components.data_cleaning import DataCleaning
        from src.components.feature_engineering import FeatureEngineering

        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        X_train, X_test, y_train, y_test = cleaner.clean_and_split(df)

        fe = FeatureEngineering()
        X_train_fe, X_test_fe, y_train, y_test = fe.engineer_features(
            X_train, X_test, y_train, y_test
        )

        preprocessor = DataPreprocessing()

        X_train_final, X_test_final, y_train, y_test = preprocessor.preprocess_data(
            X_train_fe, X_test_fe, y_train, y_test
        )

        print("Final Train Shape:", X_train_final.shape)
        print("Final Test Shape:", X_test_final.shape)

    except Exception as e:
        raise CustomException(e, sys)