from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_customer_churn.csv"
)


def load_data() -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Returns
    -------
    pd.DataFrame
        Raw customer churn dataset.
    """

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)