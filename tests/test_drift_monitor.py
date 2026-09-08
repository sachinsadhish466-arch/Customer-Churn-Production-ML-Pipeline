import pandas as pd

from src.monitoring.drift_monitor import (
    calculate_numeric_drift
)


NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend"
]


def test_numeric_drift_detected():

    reference_df = pd.DataFrame({
        "MonthlyCharges": [50, 60, 70, 80]
    })

    current_df = pd.DataFrame({
        "MonthlyCharges": [70, 80, 90, 100]
    })

    results = calculate_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        columns=["MonthlyCharges"],
        threshold=0.20
    )

    assert (
        results["MonthlyCharges"]["drift_detected"]
        is True
    )


def test_numeric_drift_not_detected():

    reference_df = pd.DataFrame({
        "MonthlyCharges": [50, 60, 70, 80]
    })

    current_df = pd.DataFrame({
        "MonthlyCharges": [51, 61, 71, 81]
    })

    results = calculate_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        columns=["MonthlyCharges"],
        threshold=0.20
    )

    assert (
        results["MonthlyCharges"]["drift_detected"]
        is False
    )


def test_multiple_numeric_features():

    reference_df = pd.DataFrame({
        "tenure": [10, 20, 30, 40],
        "MonthlyCharges": [50, 60, 70, 80]
    })

    current_df = pd.DataFrame({
        "tenure": [10, 20, 30, 40],
        "MonthlyCharges": [80, 90, 100, 110]
    })

    results = calculate_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        columns=[
            "tenure",
            "MonthlyCharges"
        ],
        threshold=0.20
    )

    assert (
        results["tenure"]["drift_detected"]
        is False
    )

    assert (
        results["MonthlyCharges"]["drift_detected"]
        is True
    )


def test_realistic_churn_numeric_features():

    reference_df = pd.DataFrame({
        "tenure": [5, 12, 24, 36, 48],
        "MonthlyCharges": [50, 60, 70, 80, 90],
        "TotalCharges": [250, 720, 1680, 2880, 4320],
        "AverageMonthlySpend": [50, 60, 70, 80, 90]
    })

    current_df = pd.DataFrame({
        "tenure": [6, 13, 25, 37, 49],
        "MonthlyCharges": [52, 62, 72, 82, 92],
        "TotalCharges": [312, 806, 1800, 3034, 4508],
        "AverageMonthlySpend": [52, 62, 72, 82, 92]
    })

    results = calculate_numeric_drift(
        reference_df=reference_df,
        current_df=current_df,
        columns=NUMERIC_FEATURES,
        threshold=0.20
    )

    assert set(results.keys()) == set(
        NUMERIC_FEATURES
    )

    for feature in NUMERIC_FEATURES:

        assert "reference_mean" in results[feature]

        assert "current_mean" in results[feature]

        assert "relative_change" in results[feature]

        assert "drift_detected" in results[feature]

        assert isinstance(
            results[feature]["drift_detected"],
            bool
        )