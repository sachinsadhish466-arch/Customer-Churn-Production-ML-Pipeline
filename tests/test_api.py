from unittest.mock import patch

import pandas as pd

from fastapi.testclient import TestClient

import api.main as api_main

from src.models.model_predictor import ChurnPredictor
from src.pipeline import ChurnPipeline

from tests.test_doubles import FakeChurnModel


# ---------------------------------------------------------
# Test Pipeline Setup
# ---------------------------------------------------------

api_main.churn_pipeline = ChurnPipeline(
    predictor=ChurnPredictor(
        model=FakeChurnModel(),
        threshold=0.5
    )
)


client = TestClient(
    api_main.app
)


# ---------------------------------------------------------
# Health Endpoint Test
# ---------------------------------------------------------

def test_health_check():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert (
        data["service"]
        == "customer-churn-prediction-api"
    )

    assert "X-Request-ID" in response.headers


# ---------------------------------------------------------
# Prediction Endpoint Test
# ---------------------------------------------------------

def test_predict_endpoint():

    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.50,
        "TotalCharges": 906.00
    }

    response = client.post(
        "/predict",
        json=customer_data
    )

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] == 1

    assert (
        data["churn_probability"]
        == 0.75
    )

    assert (
        data["prediction_label"]
        == "Likely to Churn"
    )

    assert (
        data["classification_threshold"]
        == 0.5
    )

    assert "X-Request-ID" in response.headers


# ---------------------------------------------------------
# Monitoring Endpoint Test
# ---------------------------------------------------------

def test_monitoring_endpoint():

    fake_metrics = {
        "total_predictions": 10,
        "churn_predictions": 4,
        "stay_predictions": 6,
        "churn_prediction_rate": 0.4,
        "average_churn_probability": 0.4235,
        "high_risk_predictions": 3,
        "high_risk_prediction_rate": 0.3,
        "average_latency_ms": 27.50
    }

    with patch(
        "api.main.load_prediction_logs"
    ) as mock_load_logs, patch(
        "api.main.calculate_monitoring_metrics"
    ) as mock_calculate_metrics:

        mock_load_logs.return_value = None

        mock_calculate_metrics.return_value = (
            fake_metrics
        )

        response = client.get(
            "/monitoring"
        )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["total_predictions"]
        == 10
    )

    assert (
        data["churn_predictions"]
        == 4
    )

    assert (
        data["stay_predictions"]
        == 6
    )

    assert (
        data["churn_prediction_rate"]
        == 0.4
    )

    assert (
        data["average_churn_probability"]
        == 0.4235
    )

    assert (
        data["high_risk_predictions"]
        == 3
    )

    assert (
        data["high_risk_prediction_rate"]
        == 0.3
    )

    assert (
        data["average_latency_ms"]
        == 27.50
    )

    assert "X-Request-ID" in response.headers


# ---------------------------------------------------------
# Drift Monitoring Endpoint Test
# ---------------------------------------------------------

def test_drift_endpoint():

    fake_reference_stats = pd.DataFrame([
        {
            "feature": "tenure",
            "mean": 32.37,
            "std": 24.56,
            "min": 0.0,
            "max": 72.0
        },
        {
            "feature": "MonthlyCharges",
            "mean": 64.76,
            "std": 30.09,
            "min": 18.25,
            "max": 118.75
        },
        {
            "feature": "TotalCharges",
            "mean": 2279.73,
            "std": 2266.79,
            "min": 0.0,
            "max": 8684.8
        },
        {
            "feature": "AverageMonthlySpend",
            "mean": 64.76,
            "std": 30.19,
            "min": 13.775,
            "max": 121.4
        }
    ])

    fake_current_data = pd.DataFrame({
        "tenure": [
            30,
            32,
            34,
            31,
            33
        ],

        "MonthlyCharges": [
            85,
            90,
            95,
            88,
            92
        ],

        "TotalCharges": [
            2800,
            3000,
            3200,
            2900,
            3100
        ],

        "AverageMonthlySpend": [
            85,
            90,
            95,
            88,
            92
        ]
    })

    fake_drift_results = {

        "tenure": {
            "reference_mean": 32.37,
            "current_mean": 32.0,
            "relative_change": 0.0114,
            "drift_detected": False
        },

        "MonthlyCharges": {
            "reference_mean": 64.76,
            "current_mean": 90.0,
            "relative_change": 0.3897,
            "drift_detected": True
        },

        "TotalCharges": {
            "reference_mean": 2279.73,
            "current_mean": 3000.0,
            "relative_change": 0.3159,
            "drift_detected": True
        },

        "AverageMonthlySpend": {
            "reference_mean": 64.76,
            "current_mean": 90.0,
            "relative_change": 0.3897,
            "drift_detected": True
        }
    }

    fake_report = {

        "overall_drift_detected": True,

        "total_features_checked": 4,

        "drifted_feature_count": 3,

        "drifted_features": [
            "MonthlyCharges",
            "TotalCharges",
            "AverageMonthlySpend"
        ],

        "feature_results": fake_drift_results
    }

    with patch(
        "api.main.load_reference_statistics"
    ) as mock_load_reference, patch(
        "api.main.pd.read_csv"
    ) as mock_read_csv, patch(
        "api.main.calculate_drift_from_reference_statistics"
    ) as mock_calculate_drift, patch(
        "api.main.create_drift_report"
    ) as mock_create_report:

        mock_load_reference.return_value = (
            fake_reference_stats
        )

        mock_read_csv.return_value = (
            fake_current_data
        )

        mock_calculate_drift.return_value = (
            fake_drift_results
        )

        mock_create_report.return_value = (
            fake_report
        )

        response = client.get(
            "/drift"
        )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["overall_drift_detected"]
        is True
    )

    assert (
        data["total_features_checked"]
        == 4
    )

    assert (
        data["drifted_feature_count"]
        == 3
    )

    assert set(
        data["drifted_features"]
    ) == {
        "MonthlyCharges",
        "TotalCharges",
        "AverageMonthlySpend"
    }

    assert (
        data["feature_results"]
        ["MonthlyCharges"]
        ["drift_detected"]
        is True
    )

    assert (
        data["feature_results"]
        ["tenure"]
        ["drift_detected"]
        is False
    )

    assert "X-Request-ID" in response.headers