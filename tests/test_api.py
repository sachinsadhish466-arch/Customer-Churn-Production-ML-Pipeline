from unittest.mock import patch

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
        data["average_latency_ms"]
        == 27.50
    )

    assert "X-Request-ID" in response.headers