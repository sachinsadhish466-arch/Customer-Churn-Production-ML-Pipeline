import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated customer churn data.

    Returns
    -------
    pd.DataFrame
        Cleaned customer churn data.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Standardize blank TotalCharges values
    # --------------------------------------------------

    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    # --------------------------------------------------
    # 2. Convert TotalCharges to numeric
    # --------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 3. Handle customers with zero tenure
    # --------------------------------------------------

    zero_tenure_mask = df["tenure"] == 0

    df.loc[
        zero_tenure_mask & df["TotalCharges"].isna(),
        "TotalCharges"
    ] = 0

    # --------------------------------------------------
    # 4. Final missing-value check
    # --------------------------------------------------

    if df["TotalCharges"].isna().any():
        raise ValueError(
            "Unexpected missing TotalCharges values remain "
            "after cleaning."
        )

    return df