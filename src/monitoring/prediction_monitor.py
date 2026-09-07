from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_LOG_PATH = (
    PROJECT_ROOT / "logs" / "predictions.jsonl"
)


EXPECTED_COLUMNS = [
    "timestamp",
    "request_id",
    "prediction",
    "probability",
    "classification_threshold",
    "prediction_label",
    "latency_ms"
]


def load_prediction_logs() -> pd.DataFrame:
    """
    Load prediction monitoring records from the JSONL log.
    """

    if not PREDICTION_LOG_PATH.exists():

        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    records = []

    with open(
        PREDICTION_LOG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    if not records:

        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    df = pd.DataFrame(records)

    # Keep monitoring compatible with historical
    # records created before latency monitoring existed.
    if "latency_ms" not in df.columns:

        df["latency_ms"] = pd.NA

    return df


def calculate_monitoring_metrics(
    df: pd.DataFrame
) -> dict:
    """
    Calculate basic prediction monitoring metrics.
    """

    if df.empty:

        return {
            "total_predictions": 0,
            "churn_predictions": 0,
            "stay_predictions": 0,
            "churn_prediction_rate": 0.0,
            "average_churn_probability": 0.0,
            "high_risk_predictions": 0,
            "average_latency_ms": 0.0
        }

    total_predictions = len(df)

    churn_predictions = int(
        (df["prediction"] == 1).sum()
    )

    stay_predictions = int(
        (df["prediction"] == 0).sum()
    )

    churn_prediction_rate = (
        churn_predictions / total_predictions
    )

    average_churn_probability = float(
        df["probability"].mean()
    )

    high_risk_predictions = int(
        (df["probability"] >= 0.70).sum()
    )

    if "latency_ms" in df.columns:

        latency_values = pd.to_numeric(
            df["latency_ms"],
            errors="coerce"
        ).dropna()

        if not latency_values.empty:

            average_latency_ms = float(
                latency_values.mean()
            )

        else:

            average_latency_ms = 0.0

    else:

        average_latency_ms = 0.0

    return {
        "total_predictions": total_predictions,
        "churn_predictions": churn_predictions,
        "stay_predictions": stay_predictions,
        "churn_prediction_rate": round(
            churn_prediction_rate,
            4
        ),
        "average_churn_probability": round(
            average_churn_probability,
            4
        ),
        "high_risk_predictions": high_risk_predictions,
        "average_latency_ms": round(
            average_latency_ms,
            2
        )
    }