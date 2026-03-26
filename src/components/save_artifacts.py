
# src/components/save_artifacts.py

from dataclasses import dataclass
import os
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.file_ops import save_object  # use the reusable function


@dataclass
class ArtifactSaverConfig:
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    ohe_path: str = os.path.join("artifacts", "ohe_encoder.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")


class ArtifactSaver:

    def __init__(self, config: ArtifactSaverConfig = ArtifactSaverConfig()):
        self.config = config

    def save_artifacts(self, scaler, ohe, label_encoder):
        try:
            os.makedirs("artifacts", exist_ok=True)
            save_object(scaler, self.config.scaler_path)
            logger.info(f"Scaler saved at {self.config.scaler_path}")

            save_object(ohe, self.config.ohe_path)
            logger.info(f"OHE Encoder saved at {self.config.ohe_path}")

            save_object(label_encoder, self.config.label_encoder_path)
            logger.info(f"Label Encoder saved at {self.config.label_encoder_path}")

        except Exception as e:
            logger.error("Error saving artifacts")
            raise CustomException(e, None)