from fastapi import FastAPI, HTTPException, Request
import time

import pandas as pd

from api.schemas import (
    CustomerData,
    PredictionResponse,
    MonitoringResponse
)

from src.pipeline import ChurnPipeline

from src.utils.logger import get_logger
from src.utils.request_id import generate_request_id

from src.monitoring.prediction_logger import log_prediction

from src.monitoring.prediction_monitor import (
    load_prediction_logs,
    calculate_monitoring_metrics
)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0"
)


logger = get_logger(__name__)


@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next
):

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


@app.get(
    "/monitoring",
    response_model=MonitoringResponse
)
def monitoring_metrics(
    request: Request
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.info(
        "Monitoring metrics requested | request_id=%s",
        request_id
    )

    try:

        prediction_logs = load_prediction_logs()

        metrics = calculate_monitoring_metrics(
            prediction_logs
        )

        logger.info(
            "Monitoring metrics calculated | request_id=%s | total_predictions=%s | churn_predictions=%s",
            request_id,
            metrics["total_predictions"],
            metrics["churn_predictions"]
        )

        return MonitoringResponse(
            **metrics
        )

    except Exception:

        logger.exception(
            "Monitoring metrics failed | request_id=%s",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Monitoring service encountered an internal error.",
                "request_id": request_id
            }
        )


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

        prediction_start_time = time.perf_counter()

        predictions, probabilities = (
            pipeline.predict(input_df)
        )

        latency_ms = (
            time.perf_counter()
            - prediction_start_time
        ) * 1000

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
            "Prediction completed | request_id=%s | prediction=%s | probability=%.4f | latency_ms=%.2f | label=%s",
            request_id,
            prediction,
            probability,
            latency_ms,
            prediction_label
        )

        log_prediction(
            request_id=request_id,
            prediction=prediction,
            probability=probability,
            threshold=pipeline.predictor.threshold,
            prediction_label=prediction_label,
            latency_ms=latency_ms
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