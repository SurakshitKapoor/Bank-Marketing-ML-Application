
import sys
import os
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, classification_report, make_scorer

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.file_ops import save_object


@dataclass
class ModelTrainerConfig:
    final_data_dir: str = os.path.join("data", "final")
    artifacts_dir: str = os.path.join("artifacts")
    model_path: str = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def train_model(self):

        try:

            logger.info("Model training started")

            # -----------------------------
            # Load processed data
            # -----------------------------
            X_train = pd.read_csv(
                os.path.join(self.config.final_data_dir, "X_train_final.csv")
            )

            X_test = pd.read_csv(
                os.path.join(self.config.final_data_dir, "X_test_final.csv")
            )

            y_train = pd.read_csv(
                os.path.join(self.config.final_data_dir, "y_train.csv")
            ).values.ravel()

            y_test = pd.read_csv(
                os.path.join(self.config.final_data_dir, "y_test.csv")
            ).values.ravel()

            logger.info("Processed datasets loaded")


            # -----------------------------
            # Model 1: Logistic Regression
            # -----------------------------
            log_model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )

            log_model.fit(X_train, y_train)

            log_pred = log_model.predict(X_test)

            log_recall = recall_score(y_test, log_pred, pos_label=0)

            logger.info(f"Logistic Regression Recall: {log_recall}")


            # -----------------------------
            # Model 2: Decision Tree
            # -----------------------------
            dt_model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42
            )

            dt_model.fit(X_train, y_train)

            dt_pred = dt_model.predict(X_test)

            dt_recall = recall_score(y_test, dt_pred, pos_label=0)

            logger.info(f"Decision Tree Recall: {dt_recall}")


            # -----------------------------
            # Select best model
            # -----------------------------
            if log_recall >= dt_recall:

                best_model = log_model
                best_model_name = "LogisticRegression"

                param_grid = {
                    "C": [0.01, 0.1, 1, 10]
                }

            else:

                best_model = dt_model
                best_model_name = "DecisionTree"

                param_grid = {
                    "max_depth": [5, 8, 10, 12, 15, None],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 5] }


            logger.info(f"Best base model: {best_model_name}")


            # -----------------------------
            # GridSearchCV
            # -----------------------------
            recall_0 = make_scorer(recall_score, pos_label=0)
            
            grid = GridSearchCV(
                best_model,
                param_grid,
                scoring=recall_0,
                cv=5,
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            logger.info(f"Best parameters: {grid.best_params_}")


            # -----------------------------
            # Final Evaluation
            # -----------------------------
            y_pred = best_model.predict(X_test)

            recall = recall_score(y_test, y_pred, pos_label=0)

            logger.info(f"Final Recall: {recall}")

            print("\nClassification Report:\n")
            print(classification_report(y_test, y_pred))



            # -----------------------------
            # Save model
            # -----------------------------
            save_object(
                best_model,
                self.config.model_path
                
            )

            logger.info("Best model saved successfully")

        except Exception as e:
            raise CustomException(e, sys)


# --------------------------------
# Run Training
# --------------------------------

if __name__ == "__main__":

    try:

        trainer = ModelTrainer()
        trainer.train_model()

    except Exception as e:
        raise CustomException(e, sys)