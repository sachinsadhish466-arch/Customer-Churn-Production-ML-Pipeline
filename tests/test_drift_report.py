from src.monitoring.drift_monitor import (
    create_drift_report
)


def test_drift_report_with_detected_drift():

    drift_results = {

        "tenure": {
            "reference_mean": 32.37,
            "current_mean": 33.00,
            "relative_change": 0.0195,
            "drift_detected": False
        },

        "MonthlyCharges": {
            "reference_mean": 64.76,
            "current_mean": 90.00,
            "relative_change": 0.3899,
            "drift_detected": True
        },

        "TotalCharges": {
            "reference_mean": 2279.73,
            "current_mean": 3000.00,
            "relative_change": 0.3160,
            "drift_detected": True
        }
    }

    report = create_drift_report(
        drift_results
    )

    assert (
        report["overall_drift_detected"]
        is True
    )

    assert (
        report["total_features_checked"]
        == 3
    )

    assert (
        report["drifted_feature_count"]
        == 2
    )

    assert set(
        report["drifted_features"]
    ) == {
        "MonthlyCharges",
        "TotalCharges"
    }


def test_drift_report_without_detected_drift():

    drift_results = {

        "tenure": {
            "reference_mean": 32.37,
            "current_mean": 33.00,
            "relative_change": 0.0195,
            "drift_detected": False
        },

        "MonthlyCharges": {
            "reference_mean": 64.76,
            "current_mean": 65.00,
            "relative_change": 0.0037,
            "drift_detected": False
        }
    }

    report = create_drift_report(
        drift_results
    )

    assert (
        report["overall_drift_detected"]
        is False
    )

    assert (
        report["total_features_checked"]
        == 2
    )

    assert (
        report["drifted_feature_count"]
        == 0
    )

    assert (
        report["drifted_features"]
        == []
    )


def test_empty_drift_report():

    report = create_drift_report({})

    assert (
        report["overall_drift_detected"]
        is False
    )

    assert (
        report["total_features_checked"]
        == 0
    )

    assert (
        report["drifted_feature_count"]
        == 0
    )

    assert (
        report["drifted_features"]
        == []
    )

    assert (
        report["feature_results"]
        == {}
    )