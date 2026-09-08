from typing import Dict

import pandas as pd


def calculate_numeric_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.20
) -> Dict[str, dict]:
    """
    Compare numerical feature distributions between
    reference data and current production data.

    Drift is measured using the relative change in mean.

    A feature is marked as drifted when the relative
    mean change is greater than or equal to the threshold.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference/training dataset.

    current_df : pd.DataFrame
        Current production dataset.

    columns : list[str]
        Numerical columns to monitor.

    threshold : float, default=0.20
        Relative mean-change threshold for drift detection.

    Returns
    -------
    Dict[str, dict]
        Drift results for each monitored feature.
    """

    results = {}

    for column in columns:

        if column not in reference_df.columns:

            raise ValueError(
                f"Column '{column}' not found in reference data."
            )

        if column not in current_df.columns:

            raise ValueError(
                f"Column '{column}' not found in current data."
            )

        reference_mean = float(
            reference_df[column].mean()
        )

        current_mean = float(
            current_df[column].mean()
        )

        if reference_mean == 0:

            relative_change = (
                0.0
                if current_mean == 0
                else 1.0
            )

        else:

            relative_change = abs(
                current_mean - reference_mean
            ) / abs(reference_mean)

        results[column] = {

            "reference_mean": round(
                reference_mean,
                4
            ),

            "current_mean": round(
                current_mean,
                4
            ),

            "relative_change": round(
                relative_change,
                4
            ),

            "drift_detected": (
                relative_change >= threshold
            )
        }

    return results


def calculate_drift_from_reference_statistics(
    reference_stats_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.20
) -> Dict[str, dict]:
    """
    Compare current production data against
    previously saved reference statistics.

    The reference statistics DataFrame must contain:

        feature
        mean
        std
        min
        max

    Parameters
    ----------
    reference_stats_df : pd.DataFrame
        Previously calculated reference statistics.

    current_df : pd.DataFrame
        Current production data.

    columns : list[str]
        Numerical features to monitor.

    threshold : float, default=0.20
        Relative mean-change threshold for drift detection.

    Returns
    -------
    Dict[str, dict]
        Drift results for each monitored feature.
    """

    required_columns = {
        "feature",
        "mean",
        "std",
        "min",
        "max"
    }

    missing_columns = (
        required_columns
        - set(reference_stats_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Reference statistics missing columns: "
            f"{missing_columns}"
        )

    results = {}

    for column in columns:

        if column not in reference_stats_df["feature"].values:

            raise ValueError(
                f"Feature '{column}' not found "
                f"in reference statistics."
            )

        if column not in current_df.columns:

            raise ValueError(
                f"Column '{column}' not found "
                f"in current data."
            )

        reference_row = reference_stats_df[
            reference_stats_df["feature"] == column
        ].iloc[0]

        reference_mean = float(
            reference_row["mean"]
        )

        current_mean = float(
            current_df[column].mean()
        )

        if reference_mean == 0:

            relative_change = (
                0.0
                if current_mean == 0
                else 1.0
            )

        else:

            relative_change = abs(
                current_mean - reference_mean
            ) / abs(reference_mean)

        results[column] = {

            "reference_mean": round(
                reference_mean,
                4
            ),

            "current_mean": round(
                current_mean,
                4
            ),

            "relative_change": round(
                relative_change,
                4
            ),

            "drift_detected": (
                relative_change >= threshold
            )
        }

    return results


def create_drift_report(
    drift_results: Dict[str, dict]
) -> dict:
    """
    Create an overall drift monitoring report
    from individual feature drift results.

    Parameters
    ----------
    drift_results : Dict[str, dict]
        Individual feature drift results.

    Returns
    -------
    dict
        Overall drift monitoring report.
    """

    total_features = len(
        drift_results
    )

    drifted_features = [
        feature
        for feature, result
        in drift_results.items()
        if result["drift_detected"]
    ]

    drifted_feature_count = len(
        drifted_features
    )

    overall_drift_detected = (
        drifted_feature_count > 0
    )

    return {
        "overall_drift_detected": (
            overall_drift_detected
        ),

        "total_features_checked": (
            total_features
        ),

        "drifted_feature_count": (
            drifted_feature_count
        ),

        "drifted_features": (
            drifted_features
        ),

        "feature_results": (
            drift_results
        )
    }