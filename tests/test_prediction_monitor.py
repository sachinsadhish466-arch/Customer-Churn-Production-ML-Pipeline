import pandas as pd

from src.monitoring.prediction_monitor import (
    calculate_monitoring_metrics
)


def test_calculate_monitoring_metrics():

    df = pd.DataFrame([
        {
            "timestamp": "2026-09-07T07:00:00+00:00",
            "request_id": "test-001",
            "prediction": 1,
            "probability": 0.80,
            "classification_threshold": 0.5,
            "prediction_label": "Likely to Churn",
            "latency_ms": 20.0
        },
        {
            "timestamp": "2026-09-07T07:01:00+00:00",
            "request_id": "test-002",
            "prediction": 0,
            "probability": 0.20,
            "classification_threshold": 0.5,
            "prediction_label": "Likely to Stay",
            "latency_ms": 30.0
        },
        {
            "timestamp": "2026-09-07T07:02:00+00:00",
            "request_id": "test-003",
            "prediction": 1,
            "probability": 0.90,
            "classification_threshold": 0.5,
            "prediction_label": "Likely to Churn",
            "latency_ms": 40.0
        }
    ])

    metrics = calculate_monitoring_metrics(df)

    assert metrics["total_predictions"] == 3

    assert metrics["churn_predictions"] == 2

    assert metrics["stay_predictions"] == 1

    assert metrics["churn_prediction_rate"] == 0.6667

    assert metrics["average_churn_probability"] == 0.6333

    assert metrics["high_risk_predictions"] == 2

    assert metrics["average_latency_ms"] == 30.0


def test_empty_monitoring_data():

    df = pd.DataFrame()

    metrics = calculate_monitoring_metrics(df)

    assert metrics["total_predictions"] == 0

    assert metrics["churn_predictions"] == 0

    assert metrics["stay_predictions"] == 0

    assert metrics["churn_prediction_rate"] == 0.0

    assert metrics["average_churn_probability"] == 0.0

    assert metrics["high_risk_predictions"] == 0

    assert metrics["average_latency_ms"] == 0.0


def test_monitoring_metrics_support_old_logs():

    df = pd.DataFrame([
        {
            "timestamp": "2026-09-07T07:00:00+00:00",
            "request_id": "old-log-001",
            "prediction": 1,
            "probability": 0.80,
            "classification_threshold": 0.5,
            "prediction_label": "Likely to Churn"
        }
    ])

    metrics = calculate_monitoring_metrics(df)

    assert metrics["total_predictions"] == 1

    assert metrics["average_latency_ms"] == 0.0