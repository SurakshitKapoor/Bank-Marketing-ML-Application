

from operator import le
import sys
import os
import pickle
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

from src.components.data_cleaning import DataCleaning
from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.feature_engineering import FeatureEngineering
from src.components.data_ingestion import DataIngestion
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.components.data_preprocessing import DataPreprocessing
from src.components.save_artifacts import ArtifactSaver


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def train_model(self, X_train, y_train, X_test, y_test, scaler, ohe, le):

        try:

            logger.info("Model training started")

            # Model initialization
            model = LogisticRegression(class_weight="balanced", max_iter=1000)

            logger.info("Logistic Regression model initialized")

            # Train model
            model.fit(X_train, y_train)

            logger.info("Model training completed")

            # Predictions
            y_pred = model.predict(X_test)

            logger.info("Prediction completed")

            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)

            recall = recall_score(y_test, y_pred, pos_label=0)

            logger.info(f"Model Accuracy: {accuracy}")
            logger.info(f"Model Recall: {recall}")

            # Save model
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)

            with open(self.config.trained_model_path, "wb") as file:
                pickle.dump(model, file)

            logger.info(f"Model saved at {self.config.trained_model_path}")

            # saving all preprocessing artifacts
            saver = ArtifactSaver()
            saver.save_artifacts(scaler, ohe, le)
            logger.info("All preprocessing artifacts saved successfully")
            
            return accuracy, recall

        except Exception as e:
            logger.error("Error during model training")
            raise CustomException(e, sys)


if __name__ == "__main__":

    try:

        logger.info("Model Trainer pipeline started")

        # Run full pipeline
        ingestion = DataIngestion()
        df = ingestion.load_data()

        cleaner = DataCleaning()
        clean_df = cleaner.clean_data(df)

        fe = FeatureEngineering()
        feature_df = fe.engineer_features(clean_df)

        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test, scaler, ohe, le, numerical_cols, categorical_cols = preprocessing.preprocess_data(feature_df)       

        trainer = ModelTrainer()
        
        accuracy, recall = trainer.train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            ohe,
            le
        )

        print("Accuracy:", accuracy)
        print("Recall:", recall)

    except Exception as e:
        raise CustomException(e, sys)