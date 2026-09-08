from fastapi import FastAPI, HTTPException, Request
import time

import pandas as pd

from api.schemas import (
    CustomerData,
    PredictionResponse,
    MonitoringResponse,
    DriftResponse
)

from src.pipeline import ChurnPipeline

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.request_id import generate_request_id

from src.monitoring.prediction_logger import log_prediction

from src.monitoring.prediction_monitor import (
    load_prediction_logs,
    calculate_monitoring_metrics
)

from src.monitoring.reference_statistics import (
    load_reference_statistics
)

from src.monitoring.drift_monitor import (
    calculate_drift_from_reference_statistics,
    create_drift_report
)


# ---------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------

CONFIG = load_config()


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production ML API for customer churn prediction",
    version="1.0.0"
)


logger = get_logger(__name__)


# ---------------------------------------------------------
# Request ID Middleware
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Global Pipeline
# ---------------------------------------------------------

churn_pipeline = None


def get_churn_pipeline():

    global churn_pipeline

    if churn_pipeline is None:

        logger.info(
            "Initializing churn prediction pipeline"
        )

        churn_pipeline = ChurnPipeline()

    return churn_pipeline


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    logger.info(
        "Health check requested"
    )

    return {
        "status": "healthy",
        "service": "customer-churn-prediction-api"
    }


# ---------------------------------------------------------
# Prediction Monitoring Endpoint
# ---------------------------------------------------------

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
            "Monitoring metrics calculated | "
            "request_id=%s | "
            "total_predictions=%s | "
            "churn_predictions=%s",
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
                "message": (
                    "Monitoring service encountered "
                    "an internal error."
                ),
                "request_id": request_id
            }
        )


# ---------------------------------------------------------
# Data Drift Monitoring Endpoint
# ---------------------------------------------------------

@app.get(
    "/drift",
    response_model=DriftResponse
)
def drift_metrics(
    request: Request
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.info(
        "Drift monitoring requested | request_id=%s",
        request_id
    )

    try:

        # -------------------------------------------------
        # Read monitoring configuration
        # -------------------------------------------------

        reference_statistics_path = (
            CONFIG["monitoring"]
            ["reference_statistics_path"]
        )

        current_data_path = (
            CONFIG["monitoring"]
            ["current_data_path"]
        )

        drift_threshold = float(
            CONFIG["monitoring"]
            ["drift_threshold"]
        )

        logger.info(
            "Drift configuration loaded | "
            "request_id=%s | "
            "reference=%s | "
            "current=%s | "
            "threshold=%s",
            request_id,
            reference_statistics_path,
            current_data_path,
            drift_threshold
        )

        # -------------------------------------------------
        # Load reference statistics
        # -------------------------------------------------

        reference_stats = (
            load_reference_statistics()
        )

        # -------------------------------------------------
        # Load current monitoring data
        # -------------------------------------------------

        current_df = pd.read_csv(
            current_data_path
        )

        # -------------------------------------------------
        # Numerical features being monitored
        # -------------------------------------------------

        numeric_features = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "AverageMonthlySpend"
        ]

        # -------------------------------------------------
        # Calculate feature-level drift
        # -------------------------------------------------

        drift_results = (
            calculate_drift_from_reference_statistics(
                reference_stats_df=reference_stats,
                current_df=current_df,
                columns=numeric_features,
                threshold=drift_threshold
            )
        )

        # -------------------------------------------------
        # Create overall drift report
        # -------------------------------------------------

        report = create_drift_report(
            drift_results
        )

        logger.info(
            "Drift monitoring completed | "
            "request_id=%s | "
            "overall_drift=%s | "
            "drifted_features=%s",
            request_id,
            report["overall_drift_detected"],
            report["drifted_features"]
        )

        return DriftResponse(
            **report
        )

    except Exception:

        logger.exception(
            "Drift monitoring failed | request_id=%s",
            request_id
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Drift monitoring service "
                    "encountered an internal error."
                ),
                "request_id": request_id
            }
        )


# ---------------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------------

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

        prediction_start_time = (
            time.perf_counter()
        )

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
            "Prediction completed | "
            "request_id=%s | "
            "prediction=%s | "
            "probability=%.4f | "
            "latency_ms=%.2f | "
            "label=%s",
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
                "message": (
                    "Prediction service encountered "
                    "an internal error."
                ),
                "request_id": request_id
            }
        )