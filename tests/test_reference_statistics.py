import pandas as pd
import pytest

import src.monitoring.reference_statistics as reference_statistics
from src.monitoring.reference_statistics import (
    NUMERIC_FEATURES,
    create_reference_statistics
)


@pytest.fixture
def reference_dataset(tmp_path, monkeypatch):
    test_data = pd.DataFrame({
        "tenure": [1, 12, 24, 36, 48],
        "MonthlyCharges": [30.0, 50.0, 70.0, 80.0, 100.0],
        "TotalCharges": [30.0, 600.0, 1680.0, 2880.0, 4800.0],
        "AverageMonthlySpend": [30.0, 50.0, 70.0, 80.0, 100.0]
    })

    dataset_path = tmp_path / "telco_customer_churn_features.csv"

    test_data.to_csv(
        dataset_path,
        index=False
    )

    monkeypatch.setattr(
        reference_statistics,
        "REFERENCE_DATASET_PATH",
        dataset_path
    )

    return test_data


def test_reference_statistics_structure(reference_dataset):

    stats_df = create_reference_statistics()

    assert isinstance(
        stats_df,
        pd.DataFrame
    )

    assert list(
        stats_df["feature"]
    ) == NUMERIC_FEATURES

    assert set(
        stats_df.columns
    ) == {
        "feature",
        "mean",
        "std",
        "min",
        "max"
    }


def test_reference_statistics_values(reference_dataset):

    stats_df = create_reference_statistics()

    for column in [
        "mean",
        "std",
        "min",
        "max"
    ]:

        assert (
            pd.api.types.is_numeric_dtype(
                stats_df[column]
            )
        )

    assert (
        (stats_df["min"] <= stats_df["mean"]).all()
    )

    assert (
        (stats_df["mean"] <= stats_df["max"]).all()
    )


def test_reference_statistics_features(reference_dataset):

    stats_df = create_reference_statistics()

    assert len(stats_df) == 4

    assert (
        set(stats_df["feature"])
        == set(NUMERIC_FEATURES)
    )