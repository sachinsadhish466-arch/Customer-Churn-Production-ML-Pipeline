import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready features from cleaned customer data.

    Churn is optional because it is not available
    during production inference.
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
    #    only when Churn is available
    # --------------------------------------------------

    if "Churn" in df.columns:

        df["Churn"] = df["Churn"].map({
            "No": 0,
            "Yes": 1
        })

        if df["Churn"].isna().any():
            raise ValueError(
                "Unexpected Churn values found."
            )

    # --------------------------------------------------
    # 3. Remove customer identifier
    # --------------------------------------------------

    df = df.drop(
        columns=["customerID"],
        errors="ignore"
    )

    # --------------------------------------------------
    # 4. Check engineered feature
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