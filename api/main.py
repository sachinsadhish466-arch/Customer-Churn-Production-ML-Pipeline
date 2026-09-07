from fastapi import FastAPI, HTTPException

import pandas as pd

from api.schemas import CustomerData, PredictionResponse
from src.pipeline import ChurnPipeline


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0"
)


# Load the model once when the API starts
churn_pipeline = ChurnPipeline()


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "service": "customer-churn-prediction-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerData):
    """
    Predict customer churn probability.
    """

    try:
        # Convert validated Pydantic data to dictionary
        customer_data = customer.model_dump()

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([customer_data])

        # Run prediction pipeline
        predictions, probabilities = (
            churn_pipeline.predict(input_df)
        )

        prediction = int(predictions[0])
        probability = float(probabilities[0])

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(
                probability,
                4
            ),
            prediction_label=(
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
            ),
            classification_threshold=round(
                churn_pipeline.predictor.threshold,
                4
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )