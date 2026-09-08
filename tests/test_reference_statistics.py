import pandas as pd

from src.monitoring.reference_statistics import (
    NUMERIC_FEATURES,
    create_reference_statistics
)


def test_reference_statistics_structure():

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


def test_reference_statistics_values():

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


def test_reference_statistics_features():

    stats_df = create_reference_statistics()

    assert len(stats_df) == 4

    assert (
        set(stats_df["feature"])
        == set(NUMERIC_FEATURES)
    )