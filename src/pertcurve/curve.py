"""Principal-curve fitting utilities for perturbation trajectories."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline


def principal_curve_binned(
    X,
    labels,
    control_label="control",
    n_bins=10,
    smoothing=0.5,
    n_curve_points=100,
    min_cells_per_bin=5,
):
    """Fit a binned-skeleton principal curve from control to perturbation.

    The curve is initialized by the control-to-perturbation centroid direction,
    then local bin centroids are smoothed with per-dimension splines.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_cells, n_features).")
    if len(labels) != X.shape[0]:
        raise ValueError("labels must have one entry per row of X.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one cell.")

    is_control = labels == control_label
    if not np.any(is_control):
        raise ValueError(f"No control cells found for control_label={control_label!r}.")
    if not np.any(~is_control):
        raise ValueError("At least one perturbed cell is required.")

    center_ctrl = np.mean(X[is_control], axis=0)
    center_pert = np.mean(X[~is_control], axis=0)

    direction = center_pert - center_ctrl
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.repeat(center_ctrl[None, :], n_curve_points, axis=0), np.vstack(
            [center_ctrl, center_pert]
        )

    unit_direction = direction / norm
    scores = (X - center_ctrl) @ unit_direction

    min_s, max_s = np.percentile(scores, [1, 99])
    if not np.isfinite(min_s) or not np.isfinite(max_s) or min_s == max_s:
        min_s, max_s = scores.min(), scores.max()
    if min_s == max_s:
        min_s, max_s = 0.0, norm

    bins = np.linspace(min_s, max_s, int(n_bins) + 1)
    bin_centers = [center_ctrl]
    bin_scores = [0.0]

    for s_start, s_end in zip(bins[:-1], bins[1:]):
        mask = (scores >= s_start) & (scores < s_end)
        if np.sum(mask) >= min_cells_per_bin:
            local_center = np.mean(X[mask], axis=0)
            bin_centers.append(local_center)
            bin_scores.append(float((local_center - center_ctrl) @ unit_direction))

    bin_centers.append(center_pert)
    bin_scores.append(float(norm))

    bin_scores, bin_centers = _sort_and_deduplicate(bin_scores, bin_centers)
    if len(bin_scores) < 2:
        return np.repeat(center_ctrl[None, :], n_curve_points, axis=0), bin_centers

    t_smooth = np.linspace(bin_scores[0], bin_scores[-1], int(n_curve_points))
    curve = np.zeros((len(t_smooth), X.shape[1]), dtype=float)

    for dim in range(X.shape[1]):
        if len(bin_scores) < 4:
            curve[:, dim] = np.interp(t_smooth, bin_scores, bin_centers[:, dim])
            continue

        weights = np.ones(len(bin_scores), dtype=float)
        weights[0] = 3.0
        weights[-1] = 3.0
        k = min(3, len(bin_scores) - 1)
        try:
            spline = UnivariateSpline(
                bin_scores,
                bin_centers[:, dim],
                w=weights,
                k=k,
                s=float(smoothing) * len(bin_scores),
            )
            curve[:, dim] = spline(t_smooth)
        except Exception:
            curve[:, dim] = np.interp(t_smooth, bin_scores, bin_centers[:, dim])

    return curve, bin_centers


def _sort_and_deduplicate(scores, centers):
    scores = np.asarray(scores, dtype=float)
    centers = np.asarray(centers, dtype=float)
    order = np.argsort(scores)
    scores = scores[order]
    centers = centers[order]

    unique_scores = []
    unique_centers = []
    for score in np.unique(scores):
        mask = scores == score
        unique_scores.append(score)
        unique_centers.append(np.mean(centers[mask], axis=0))

    return np.asarray(unique_scores, dtype=float), np.asarray(unique_centers, dtype=float)
