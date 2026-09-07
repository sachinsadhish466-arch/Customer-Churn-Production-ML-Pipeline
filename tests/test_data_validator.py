import pandas as pd
import pytest

from src.data.data_validator import validate_data


def test_valid_customer_data():

    df = pd.DataFrame([
        {
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
    ])

    assert validate_data(
        df,
        require_target=False
    ) is True


def test_invalid_negative_tenure():

    df = pd.DataFrame([
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": -1,
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
    ])

    with pytest.raises(ValueError):

        validate_data(
            df,
            require_target=False
        )