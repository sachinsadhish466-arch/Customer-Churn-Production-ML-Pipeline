from fastapi.testclient import TestClient

import api.main as api_main

from src.models.model_predictor import ChurnPredictor
from src.pipeline import ChurnPipeline

from tests.test_doubles import FakeChurnModel


api_main.churn_pipeline = ChurnPipeline(
    predictor=ChurnPredictor(
        model=FakeChurnModel(),
        threshold=0.5
    )
)


client = TestClient(
    api_main.app
)


def test_health_endpoint():

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


def test_prediction_endpoint():

    customer_data = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "No",

        "Dependents": "No",

        "tenure": 12,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "DSL",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "No",

        "StreamingMovies": "No",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 70.0,

        "TotalCharges": 840.0
    }


    response = client.post(
        "/predict",
        json=customer_data
    )


    assert response.status_code == 200


    data = response.json()


    assert "churn_prediction" in data

    assert "churn_probability" in data

    assert "prediction_label" in data

    assert "classification_threshold" in data


    assert (
        data["churn_prediction"]
        in [0, 1]
    )


    assert (
        0
        <= data["churn_probability"]
        <= 1
    )


    assert (
        0
        <= data["classification_threshold"]
        <= 1
    )