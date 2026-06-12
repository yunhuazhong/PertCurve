"""High-level perturbation scoring and distance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance

from .curve import principal_curve_binned
from .projection import (
    get_polyline_cumulative_length,
    normalize_arc_lengths_unclipped,
    project_points_to_curve_continuous,
)


def score_perturbation(
    X,
    labels,
    cell_ids=None,
    control_label="control",
    n_bins=20,
    smoothing=0.5,
    n_curve_points=100,
):
    """Fit a PertCurve trajectory and return per-cell scores plus summary stats."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if cell_ids is None:
        cell_ids = np.arange(X.shape[0]).astype(str)

    curve, bin_centers = principal_curve_binned(
        X,
        labels,
        control_label=control_label,
        n_bins=n_bins,
        smoothing=smoothing,
        n_curve_points=n_curve_points,
    )
    projected, arc_lengths, distances = project_points_to_curve_continuous(X, curve)
    scores, control_arc, perturb_arc = normalize_arc_lengths_unclipped(
        arc_lengths, X, labels, curve, control_label=control_label
    )

    result = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "perturbation": labels,
            "normalized_pseudotime": scores,
            "arc_length": arc_lengths,
            "projection_distance": distances,
        }
    )
    if X.shape[1] >= 2:
        result["original_PC1"] = X[:, 0]
        result["original_PC2"] = X[:, 1]
        result["projection_PC1"] = projected[:, 0]
        result["projection_PC2"] = projected[:, 1]

    stats = compute_distance_stats(
        arc_lengths=arc_lengths,
        labels=labels,
        curve=curve,
        control_label=control_label,
    )
    stats["control_arc"] = control_arc
    stats["perturbation_arc"] = perturb_arc

    return result, stats, curve, bin_centers


def compute_distance_stats(arc_lengths, labels, curve, control_label="control", n_hist_bins=50):
    """Compute trajectory distance statistics between control and perturbed cells."""
    arc_lengths = np.asarray(arc_lengths, dtype=float)
    labels = np.asarray(labels)
    is_control = labels == control_label
    ctrl = arc_lengths[is_control]
    pert = arc_lengths[~is_control]

    if len(ctrl) == 0 or len(pert) == 0:
        raise ValueError("Both control and perturbed cells are required.")

    wasserstein = wasserstein_distance(ctrl, pert)
    kl = _kl_divergence(pert, ctrl, n_bins=n_hist_bins)
    curve_length = get_polyline_cumulative_length(curve)[-1]

    return {
        "wasserstein_dist": float(wasserstein),
        "kl_divergence": float(kl),
        "curve_length": float(curve_length),
        "mean_pseudotime_shift": float(np.mean(pert) - np.mean(ctrl)),
        "curve_euclidean_dist": float(np.linalg.norm(curve[0] - curve[-1])),
        "n_cells_pert": int(len(pert)),
        "n_cells_ctrl": int(len(ctrl)),
    }


def _kl_divergence(pert, ctrl, n_bins=50, epsilon=1e-10):
    lower = min(float(np.min(pert)), float(np.min(ctrl)))
    upper = max(float(np.max(pert)), float(np.max(ctrl)))
    if upper == lower:
        upper = lower + 1e-5

    bins = np.linspace(lower, upper, int(n_bins) + 1)
    hist_pert, _ = np.histogram(pert, bins=bins, density=False)
    hist_ctrl, _ = np.histogram(ctrl, bins=bins, density=False)
    hist_pert = hist_pert.astype(float) + epsilon
    hist_ctrl = hist_ctrl.astype(float) + epsilon
    hist_pert /= hist_pert.sum()
    hist_ctrl /= hist_ctrl.sum()
    return entropy(hist_pert, hist_ctrl)
