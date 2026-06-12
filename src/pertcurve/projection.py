"""Polyline projection and pseudotime normalization utilities."""

from __future__ import annotations

import numpy as np


def get_polyline_cumulative_length(curve):
    """Return cumulative arc length at each polyline node."""
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2:
        raise ValueError("curve must be a 2D array.")
    if len(curve) == 0:
        return np.array([], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def project_points_to_curve_continuous(X, curve):
    """Project points to their nearest locations on a polyline."""
    X = np.asarray(X, dtype=float)
    curve = np.asarray(curve, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if curve.ndim != 2:
        raise ValueError("curve must be a 2D array.")
    if curve.shape[1] != X.shape[1]:
        raise ValueError("X and curve must have the same feature dimension.")
    if curve.shape[0] < 2:
        raise ValueError("curve must contain at least two points.")

    n_samples = X.shape[0]
    best_proj_coords = np.zeros_like(X, dtype=float)
    best_arc_lengths = np.zeros(n_samples, dtype=float)
    min_dist_sq = np.full(n_samples, np.inf, dtype=float)
    node_cum_lengths = get_polyline_cumulative_length(curve)

    for idx in range(curve.shape[0] - 1):
        start = curve[idx]
        end = curve[idx + 1]
        segment = end - start
        segment_len_sq = float(np.sum(segment**2))

        if segment_len_sq > 1e-12:
            t = ((X - start) @ segment) / segment_len_sq
            t = np.clip(t, 0.0, 1.0)
            current_proj = start + t[:, None] * segment
        else:
            t = np.zeros(n_samples, dtype=float)
            current_proj = np.repeat(start[None, :], n_samples, axis=0)

        dist_sq = np.sum((X - current_proj) ** 2, axis=1)
        closer = dist_sq < min_dist_sq
        min_dist_sq[closer] = dist_sq[closer]
        best_proj_coords[closer] = current_proj[closer]
        best_arc_lengths[closer] = node_cum_lengths[idx] + t[closer] * (
            node_cum_lengths[idx + 1] - node_cum_lengths[idx]
        )

    return best_proj_coords, best_arc_lengths, np.sqrt(min_dist_sq)


def normalize_arc_lengths_unclipped(
    arc_lengths,
    X,
    labels,
    curve,
    control_label="control",
    eps=1e-6,
):
    """Normalize arc lengths so control centroid is 0 and perturbed centroid is 1."""
    arc_lengths = np.asarray(arc_lengths, dtype=float)
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    is_control = labels == control_label
    if not np.any(is_control):
        raise ValueError(f"No control cells found for control_label={control_label!r}.")
    if not np.any(~is_control):
        raise ValueError("At least one perturbed cell is required.")

    center_control = np.mean(X[is_control], axis=0)
    center_perturb = np.mean(X[~is_control], axis=0)

    _, center_arcs, _ = project_points_to_curve_continuous(
        np.vstack([center_control, center_perturb]), curve
    )
    l_ctrl, l_pert = float(center_arcs[0]), float(center_arcs[1])
    denom = l_pert - l_ctrl

    if abs(denom) < eps:
        denom = eps if denom >= 0 else -eps

    return (arc_lengths - l_ctrl) / denom, l_ctrl, l_pert
