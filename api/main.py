from fastapi import FastAPI, HTTPException

import pandas as pd

from api.schemas import (
    CustomerData,
    PredictionResponse
)

from src.pipeline import ChurnPipeline
from src.utils.logger import get_logger


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0"
)


logger = get_logger(__name__)


churn_pipeline = None


def get_churn_pipeline():

    global churn_pipeline

    if churn_pipeline is None:
        logger.info("Initializing churn prediction pipeline")
        churn_pipeline = ChurnPipeline()

    return churn_pipeline


@app.get("/health")
def health_check():

    logger.info("Health check requested")

    return {
        "status": "healthy",
        "service": "customer-churn-prediction-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerData):

    logger.info("Prediction request received")

    try:

        pipeline = get_churn_pipeline()

        customer_data = customer.model_dump()

        input_df = pd.DataFrame(
            [customer_data]
        )

        predictions, probabilities = (
            pipeline.predict(input_df)
        )

        prediction = int(
            predictions[0]
        )

        probability = float(
            probabilities[0]
        )

        prediction_label = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        logger.info(
            "Prediction completed | prediction=%s | probability=%.4f | label=%s",
            prediction,
            probability,
            prediction_label
        )

        return PredictionResponse(

            churn_prediction=prediction,

            churn_probability=round(
                probability,
                4
            ),

            prediction_label=prediction_label,

            classification_threshold=round(
                pipeline.predictor.threshold,
                4
            )
        )

    except Exception as error:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )