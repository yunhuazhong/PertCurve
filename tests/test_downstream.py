import numpy as np

from pertcurve.downstream import (
    classify_hill_response,
    fit_hill_response,
    hurdle_trend,
    response_efficiency,
)


def test_hurdle_trend_reports_detection_and_expected_expression():
    scores = np.linspace(0, 1, 100)
    expr = np.where(scores > 0.4, scores, 0.0)

    trend = hurdle_trend(scores, expr, n_bins=5, min_cells=5)

    assert len(trend) == 5
    assert trend["detection_probability"].between(0, 1).all()
    assert trend["expected_expression"].iloc[-1] > trend["expected_expression"].iloc[0]


def test_hill_fit_classifies_increasing_response():
    rng = np.random.default_rng(0)
    scores = np.linspace(0.02, 1.0, 120)
    expr = 0.2 + 1.5 / (1.0 + (0.45 / scores) ** 3.0)
    expr = expr + rng.normal(scale=0.02, size=len(scores))

    fit = fit_hill_response(scores, expr)

    assert fit["rss"] < 0.2
    assert fit["response_type"].endswith("_up")
    assert response_efficiency(2.0, fit["lower"], fit["upper"]) > 0


def test_hill_classifier_handles_flat_response():
    assert classify_hill_response(1.0, 0.5, 0.5, 0.5) == "flat"
