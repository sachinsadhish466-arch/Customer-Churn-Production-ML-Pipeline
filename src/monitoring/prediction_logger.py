from pathlib import Path
from datetime import datetime, timezone
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"


def log_prediction(
    request_id: str,
    prediction: int,
    probability: float,
    threshold: float,
    prediction_label: str,
    latency_ms: float
) -> None:
    """
    Store a single prediction event as a JSON Lines record.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "prediction": int(prediction),
        "probability": float(probability),
        "classification_threshold": float(threshold),
        "prediction_label": prediction_label,
        "latency_ms": round(float(latency_ms), 2)
    }

    with open(
        PREDICTION_LOG_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record) + "\n"
        )