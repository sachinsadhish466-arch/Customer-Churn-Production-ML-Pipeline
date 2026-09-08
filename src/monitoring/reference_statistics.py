from pathlib import Path

import pandas as pd

from src.utils.config_loader import load_config


# ---------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG = load_config()


# ---------------------------------------------------------
# Monitoring Configuration
# ---------------------------------------------------------

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / CONFIG["monitoring"]["reference_statistics_path"]
)

REFERENCE_STATS_PATH = (
    PROJECT_ROOT
    / CONFIG["monitoring"]["reference_statistics_path"]
)


# ---------------------------------------------------------
# Reference Dataset
# ---------------------------------------------------------

# The source dataset used to calculate the reference
# statistics remains the processed feature dataset.

REFERENCE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)


# ---------------------------------------------------------
# Numerical Features
# ---------------------------------------------------------

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend"
]


# ---------------------------------------------------------
# Create Reference Statistics
# ---------------------------------------------------------

def create_reference_statistics() -> pd.DataFrame:
    """
    Calculate reference statistics from the
    training/reference dataset.
    """

    if not REFERENCE_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Reference dataset not found: "
            f"{REFERENCE_DATASET_PATH}"
        )

    df = pd.read_csv(
        REFERENCE_DATASET_PATH
    )

    missing_columns = [
        column
        for column in NUMERIC_FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing numeric features: "
            f"{missing_columns}"
        )

    statistics = []

    for column in NUMERIC_FEATURES:

        statistics.append({

            "feature": column,

            "mean": float(
                df[column].mean()
            ),

            "std": float(
                df[column].std()
            ),

            "min": float(
                df[column].min()
            ),

            "max": float(
                df[column].max()
            )
        })

    stats_df = pd.DataFrame(
        statistics
    )

    return stats_df


# ---------------------------------------------------------
# Save Reference Statistics
# ---------------------------------------------------------

def save_reference_statistics() -> None:
    """
    Generate and save reference statistics.
    """

    stats_df = create_reference_statistics()

    REFERENCE_STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stats_df.to_csv(
        REFERENCE_STATS_PATH,
        index=False
    )


# ---------------------------------------------------------
# Load Reference Statistics
# ---------------------------------------------------------

def load_reference_statistics() -> pd.DataFrame:
    """
    Load previously saved reference statistics.
    """

    if not REFERENCE_STATS_PATH.exists():

        raise FileNotFoundError(
            f"Reference statistics not found: "
            f"{REFERENCE_STATS_PATH}"
        )

    stats_df = pd.read_csv(
        REFERENCE_STATS_PATH
    )

    required_columns = {
        "feature",
        "mean",
        "std",
        "min",
        "max"
    }

    missing_columns = (
        required_columns
        - set(stats_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Reference statistics missing "
            f"columns: {missing_columns}"
        )

    missing_features = [
        feature
        for feature in NUMERIC_FEATURES
        if feature not in stats_df["feature"].values
    ]

    if missing_features:

        raise ValueError(
            f"Reference statistics missing "
            f"features: {missing_features}"
        )

    return stats_df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    save_reference_statistics()

    print(
        "Reference statistics saved to:"
    )

    print(
        REFERENCE_STATS_PATH
    )