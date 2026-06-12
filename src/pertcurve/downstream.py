"""Downstream response modeling along PertCurve pseudotime."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def hurdle_trend(
    scores,
    expression,
    n_bins=20,
    min_cells=10,
    nonzero_threshold=0.0,
):
    """Estimate a zero-inflation-aware expression trend over pseudotime bins."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(expression, dtype=float)
    if scores.ndim != 1 or y.ndim != 1:
        raise ValueError("scores and expression must be 1D.")
    if len(scores) != len(y):
        raise ValueError("scores and expression must have the same length.")

    bins = np.linspace(np.nanmin(scores), np.nanmax(scores), int(n_bins) + 1)
    if bins[0] == bins[-1]:
        bins = np.linspace(0.0, 1.0, int(n_bins) + 1)

    rows = []
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == len(bins) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n < min_cells:
            continue

        values = y[mask]
        detected = values > nonzero_threshold
        detection_prob = float(np.mean(detected))
        nonzero_mean = float(np.mean(values[detected])) if np.any(detected) else 0.0
        rows.append(
            {
                "bin": idx,
                "score_midpoint": float((lo + hi) / 2.0),
                "n_cells": n,
                "detection_probability": detection_prob,
                "nonzero_mean": nonzero_mean,
                "expected_expression": detection_prob * nonzero_mean,
            }
        )
    return pd.DataFrame(rows)


def hill_function(t, slope, lower, upper, ec50):
    """Four-parameter log-logistic response curve."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    ec50 = max(float(ec50), 1e-6)
    return lower + (upper - lower) / (1.0 + (ec50 / t) ** slope)


def fit_hill_response(scores, expression, maxfev=10000):
    """Fit a Hill curve and return parameters plus residual diagnostics."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(expression, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(y)
    scores = np.clip(scores[mask], 1e-6, 1.0)
    y = y[mask]
    if len(scores) < 5:
        return _hill_result(np.nan, np.nan, np.nan, np.nan, np.nan, "too_few_cells")

    lower0 = float(np.nanpercentile(y, 10))
    upper0 = float(np.nanpercentile(y, 90))
    if abs(upper0 - lower0) < 1e-8:
        return _hill_result(1.0, lower0, upper0, 0.5, 0.0, "flat")

    try:
        params, _ = curve_fit(
            hill_function,
            scores,
            y,
            p0=[2.0, lower0, upper0, 0.5],
            bounds=([0.05, -np.inf, -np.inf, 1e-4], [20.0, np.inf, np.inf, 1.0]),
            maxfev=maxfev,
        )
        pred = hill_function(scores, *params)
        rss = float(np.sum((y - pred) ** 2))
        trend_type = classify_hill_response(*params)
        return _hill_result(*params, rss, trend_type)
    except (RuntimeError, ValueError, FloatingPointError):
        return _hill_result(np.nan, np.nan, np.nan, np.nan, np.nan, "fit_failed")


def classify_hill_response(slope, lower, upper, ec50):
    """Map fitted Hill parameters to manuscript response archetypes."""
    if not np.all(np.isfinite([slope, lower, upper, ec50])):
        return "unclassified"
    amplitude = upper - lower
    if abs(amplitude) < 1e-6:
        return "flat"
    if ec50 <= 0.33:
        prefix = "sensitive"
    elif ec50 >= 0.67:
        prefix = "threshold"
    else:
        prefix = "proportional" if slope <= 2.0 else "switch_like"
    direction = "up" if amplitude > 0 else "down"
    return f"{prefix}_{direction}"


def response_efficiency(wasserstein, lower, upper, delta=1e-6):
    """Rank perturbations by distributional shift per fitted dynamic range."""
    dynamic_range = abs(float(upper) - float(lower))
    return float(wasserstein) / (dynamic_range + float(delta))


def summarize_gene_response(scores, expression, wasserstein=None, n_bins=20):
    """Fit hurdle and Hill summaries for one downstream gene."""
    trend = hurdle_trend(scores, expression, n_bins=n_bins)
    hill = fit_hill_response(scores, expression)
    if wasserstein is not None and np.isfinite(hill["lower"]) and np.isfinite(hill["upper"]):
        hill["response_efficiency"] = response_efficiency(
            wasserstein,
            hill["lower"],
            hill["upper"],
        )
    else:
        hill["response_efficiency"] = np.nan
    return trend, hill


def _hill_result(slope, lower, upper, ec50, rss, trend_type):
    return {
        "hill_slope": float(slope) if np.isfinite(slope) else np.nan,
        "lower": float(lower) if np.isfinite(lower) else np.nan,
        "upper": float(upper) if np.isfinite(upper) else np.nan,
        "ec50": float(ec50) if np.isfinite(ec50) else np.nan,
        "rss": float(rss) if np.isfinite(rss) else np.nan,
        "response_type": trend_type,
    }
