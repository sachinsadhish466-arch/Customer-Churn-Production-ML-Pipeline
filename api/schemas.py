from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(ge=0, le=1)
    churn_probability: float = Field(ge=0, le=1)
    prediction_label: str
    classification_threshold: float = Field(ge=0, le=1)


class MonitoringResponse(BaseModel):
    total_predictions: int
    churn_predictions: int
    stay_predictions: int
    churn_prediction_rate: float = Field(ge=0, le=1)
    average_churn_probability: float = Field(ge=0, le=1)
    high_risk_predictions: int
    average_latency_ms: float = Field(ge=0)