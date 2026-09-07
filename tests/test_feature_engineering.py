import pandas as pd

from src.features.feature_engineering import create_features


def test_average_monthly_spend():

    df = pd.DataFrame([
        {
            "customerID": "TEST001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 10,
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
            "MonthlyCharges": 50.0,
            "TotalCharges": 500.0
        }
    ])

    result = create_features(df)

    assert "AverageMonthlySpend" in result.columns

    assert result["AverageMonthlySpend"].iloc[0] == 50.0


def test_zero_tenure_average_spend():

    df = pd.DataFrame([
        {
            "customerID": "TEST002",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
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
            "MonthlyCharges": 35.0,
            "TotalCharges": 0.0
        }
    ])

    result = create_features(df)

    assert result["AverageMonthlySpend"].iloc[0] == 35.0


def test_customer_id_removed():

    df = pd.DataFrame([
        {
            "customerID": "TEST003",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
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
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 60.0,
            "TotalCharges": 720.0
        }
    ])

    result = create_features(df)

    assert "customerID" not in result.columns