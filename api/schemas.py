from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """
    Input schema for customer churn prediction.
    """

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
    """
    Response returned by the churn prediction API.
    """

    churn_prediction: int = Field(ge=0, le=1)
    churn_probability: float = Field(ge=0, le=1)
    prediction_label: str
    classification_threshold: float = Field(ge=0, le=1)