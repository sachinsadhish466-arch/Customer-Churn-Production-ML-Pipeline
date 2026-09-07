import pandas as pd

from src.models.model_predictor import ChurnPredictor


def test_model_loads():

    predictor = ChurnPredictor()

    assert predictor.model is not None
    assert predictor.threshold is not None


def test_prediction_probability():

    predictor = ChurnPredictor()

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
            "TotalCharges": 840.0,
            "AverageMonthlySpend": 70.0
        }
    ])

    probability = predictor.predict_probability(df)

    assert len(probability) == 1
    assert 0 <= probability[0] <= 1


def test_prediction_output():

    predictor = ChurnPredictor()

    df = pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 90.0,
            "TotalCharges": 2160.0,
            "AverageMonthlySpend": 90.0
        }
    ])

    predictions = predictor.predict(df)

    assert len(predictions) == 1
    assert predictions[0] in [0, 1]