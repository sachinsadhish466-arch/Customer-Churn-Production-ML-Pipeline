from fastapi import FastAPI, HTTPException, Request

import pandas as pd

from api.schemas import (
    CustomerData,
    PredictionResponse
)

from src.pipeline import ChurnPipeline
from src.utils.logger import get_logger
from src.utils.request_id import generate_request_id


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0"
)


logger = get_logger(__name__)


@app.middleware("http")
async def add_request_id(request: Request, call_next):

    request_id = generate_request_id()

    request.state.request_id = request_id

    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path
    )

    try:

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed | request_id=%s | status_code=%s",
            request_id,
            response.status_code
        )

        return response

    except Exception:

        logger.exception(
            "Request failed | request_id=%s",
            request_id
        )

        raise


churn_pipeline = None


def get_churn_pipeline():

    global churn_pipeline

    if churn_pipeline is None:

        logger.info(
            "Initializing churn prediction pipeline"
        )

        churn_pipeline = ChurnPipeline()

    return churn_pipeline


@app.get("/health")
def health_check():

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy",
        "service": "customer-churn-prediction-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(
    request: Request,
    customer: CustomerData
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.info(
        "Prediction request received | request_id=%s",
        request_id
    )

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
            "Prediction completed | request_id=%s | prediction=%s | probability=%.4f | label=%s",
            request_id,
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

    except Exception:

        logger.exception(
            "Prediction failed | request_id=%s",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction service encountered an internal error.",
                "request_id": request_id
            }
        )