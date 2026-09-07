from pathlib import Path

import joblib
import pandas as pd

from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config()

MODEL_PATH = PROJECT_ROOT / CONFIG["model"]["path"]
THRESHOLD_PATH = PROJECT_ROOT / CONFIG["model"]["threshold_path"]


class ChurnPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold_path: Path = THRESHOLD_PATH,
    ):
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.model = self._load_model()
        self.threshold = self._load_threshold()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        return joblib.load(self.model_path)

    def _load_threshold(self):
        if not self.threshold_path.exists():
            raise FileNotFoundError(
                f"Threshold file not found: {self.threshold_path}"
            )

        return float(joblib.load(self.threshold_path))

    def predict_probability(self, X: pd.DataFrame):
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict(self, X: pd.DataFrame):
        probabilities = self.predict_probability(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions

    def predict_with_probability(self, X: pd.DataFrame):
        probabilities = self.predict_probability(X)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions, probabilities