from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "simulated_production_batch.csv"
)


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend"
]


def create_simulated_production_batch() -> pd.DataFrame:
    """
    Create a simulated production batch with
    intentionally shifted customer behavior.

    The batch is based on the reference dataset,
    but selected numerical features are shifted
    to simulate production data drift.
    """

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: "
            f"{REFERENCE_DATA_PATH}"
        )

    df = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    # Take a deterministic sample so the
    # generated batch is reproducible.
    batch = (
        df
        .sample(
            n=500,
            random_state=42
        )
        .copy()
        .reset_index(drop=True)
    )

    # -------------------------------------------------
    # Simulate production behavior changes
    # -------------------------------------------------

    # Increase monthly charges by 35%.
    batch["MonthlyCharges"] = (
        batch["MonthlyCharges"] * 1.35
    )

    # Increase average monthly spending by 35%.
    batch["AverageMonthlySpend"] = (
        batch["AverageMonthlySpend"] * 1.35
    )

    # Keep tenure unchanged to demonstrate that
    # not every feature necessarily experiences drift.
    #
    # Recalculate TotalCharges consistently with
    # the shifted monthly spending.
    batch["TotalCharges"] = (
        batch["AverageMonthlySpend"]
        * batch["tenure"]
    )

    # Handle zero-tenure customers using the same
    # business rule as the feature-engineering pipeline.
    zero_tenure_mask = (
        batch["tenure"] == 0
    )

    batch.loc[
        zero_tenure_mask,
        "TotalCharges"
    ] = 0

    return batch


def save_simulated_production_batch() -> None:
    """
    Generate and save the simulated production batch.
    """

    batch = create_simulated_production_batch()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    batch.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "Simulated production batch saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"Rows: {len(batch)}"
    )

    print(
        f"Columns: {len(batch.columns)}"
    )


if __name__ == "__main__":

    save_simulated_production_batch()