import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from cleaned customer data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned customer churn data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing engineered features and target.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Create AverageMonthlySpend
    # --------------------------------------------------

    df["AverageMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # --------------------------------------------------
    # 2. Convert Churn into binary target
    # --------------------------------------------------

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # --------------------------------------------------
    # 3. Validate target conversion
    # --------------------------------------------------

    if df["Churn"].isna().any():
        raise ValueError(
            "Unexpected Churn values found."
        )

    # --------------------------------------------------
    # 4. Remove customer identifier
    # --------------------------------------------------

    df = df.drop(
        columns=["customerID"],
        errors="ignore"
    )

    # --------------------------------------------------
    # 5. Check engineered feature
    # --------------------------------------------------

    if not np.isfinite(
        df["AverageMonthlySpend"]
    ).all():
        raise ValueError(
            "Invalid values found in AverageMonthlySpend."
        )

    return df


def split_features_target(
    df: pd.DataFrame
):
    """
    Separate input features from the target.

    Returns
    -------
    X : pd.DataFrame
        Model input features.

    y : pd.Series
        Binary churn target.
    """

    if "Churn" not in df.columns:
        raise ValueError(
            "Target column 'Churn' not found."
        )

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"]

    return X, y