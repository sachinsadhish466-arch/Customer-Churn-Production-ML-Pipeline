import pandas as pd


REQUIRED_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def validate_schema(
    df: pd.DataFrame,
    require_target: bool = False
) -> None:
    """
    Validate that the required columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_FEATURE_COLUMNS
        if column not in df.columns
    ]

    if require_target and "Churn" not in df.columns:
        missing_columns.append("Churn")

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_data_quality(df: pd.DataFrame) -> None:
    """
    Validate basic data-quality rules.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    if (df["tenure"] < 0).any():
        raise ValueError(
            "Invalid tenure values detected."
        )

    if (df["MonthlyCharges"] < 0).any():
        raise ValueError(
            "Invalid MonthlyCharges values detected."
        )

    if (df["TotalCharges"] < 0).any():
        raise ValueError(
            "Invalid TotalCharges values detected."
        )


def validate_data(
    df: pd.DataFrame,
    require_target: bool = False
) -> bool:
    """
    Run all data validation checks.
    """

    validate_schema(
        df,
        require_target=require_target
    )

    validate_data_quality(df)

    return True