"""Trajectory quality metrics used to evaluate PertCurve scores."""

from __future__ import annotations

import numpy as np


def balanced_subsample_indices(
    labels,
    control_label="control",
    max_per_group=None,
    random_state=0,
):
    """Return indices after balancing control and perturbed cell counts.

    The function is intended for one control-vs-one-perturbation subset. If a
    subset contains multiple non-control labels, all non-control cells are
    treated as one perturbed group.
    """
    labels = np.asarray(labels).astype(str)
    control_label = str(control_label)
    rng = np.random.default_rng(random_state)

    ctrl_idx = np.flatnonzero(labels == control_label)
    pert_idx = np.flatnonzero(labels != control_label)
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Both control and perturbed cells are required.")

    n = min(len(ctrl_idx), len(pert_idx))
    if max_per_group is not None:
        n = min(n, int(max_per_group))
    if n <= 0:
        raise ValueError("Balanced sample size must be positive.")

    ctrl_take = rng.choice(ctrl_idx, size=n, replace=False)
    pert_take = rng.choice(pert_idx, size=n, replace=False)
    out = np.concatenate([ctrl_take, pert_take])
    rng.shuffle(out)
    return out


def smoothness_score(scores, expression):
    """Measure local expression roughness after ordering cells by pseudotime.

    Lower values indicate smoother downstream variation along the trajectory.
    Returns one value per expression feature.
    """
    scores = _as_1d(scores, "scores")
    X = _as_2d(expression, "expression")
    _check_same_rows(scores, X)
    order = np.argsort(scores)
    X_sorted = X[order]
    if X_sorted.shape[0] < 2:
        return np.full(X_sorted.shape[1], np.nan)

    local_var = np.mean(np.diff(X_sorted, axis=0) ** 2, axis=0)
    global_var = np.var(X_sorted, axis=0)
    return local_var / np.maximum(global_var, 1e-12)


def reconstruct_by_sliding_window(scores, expression, window_size=None, window_fraction=0.1):
    """Approximate a trajectory reconstruction with a local smoothing window."""
    scores = _as_1d(scores, "scores")
    X = _as_2d(expression, "expression")
    _check_same_rows(scores, X)
    n_cells = X.shape[0]
    if n_cells == 0:
        return np.empty_like(X)

    if window_size is None:
        window_size = max(3, int(round(n_cells * float(window_fraction))))
    window_size = max(1, min(int(window_size), n_cells))
    half = window_size // 2

    order = np.argsort(scores)
    X_sorted = X[order]
    recon_sorted = np.zeros_like(X_sorted, dtype=float)
    for i in range(n_cells):
        lo = max(0, i - half)
        hi = min(n_cells, i + half + 1)
        recon_sorted[i] = np.mean(X_sorted[lo:hi], axis=0)

    recon = np.zeros_like(recon_sorted)
    recon[order] = recon_sorted
    return recon


def trajectory_reconstruction_mse(scores, expression, window_size=None, window_fraction=0.1):
    """Return MSE between expression and the smoothed 1D trajectory estimate."""
    X = _as_2d(expression, "expression")
    recon = reconstruct_by_sliding_window(
        scores,
        X,
        window_size=window_size,
        window_fraction=window_fraction,
    )
    return float(np.mean((X - recon) ** 2))


def pseudotime_mutual_information(scores, expression, n_neighbors=5, random_state=0):
    """Estimate MI between pseudotime and each expression feature."""
    from sklearn.feature_selection import mutual_info_regression

    scores = _as_1d(scores, "scores")
    X = _as_2d(expression, "expression")
    _check_same_rows(scores, X)
    score_feature = scores.reshape(-1, 1)
    values = []
    for idx in range(X.shape[1]):
        values.append(
            mutual_info_regression(
                score_feature,
                X[:, idx],
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        )
    return np.asarray(values, dtype=float)


def neighbor_jaccard(scores, embedding, n_neighbors=15):
    """Compare k-nearest-neighbor sets in high-dimensional and 1D spaces."""
    from sklearn.neighbors import NearestNeighbors

    scores = _as_1d(scores, "scores")
    X = _as_2d(embedding, "embedding")
    _check_same_rows(scores, X)
    n_cells = X.shape[0]
    if n_cells <= 1:
        return np.nan
    k = max(1, min(int(n_neighbors), n_cells - 1))

    hd = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)
    one_d = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(scores.reshape(-1, 1))
        .kneighbors(return_distance=False)
    )

    values = []
    for i in range(n_cells):
        a = set(hd[i][hd[i] != i][:k])
        b = set(one_d[i][one_d[i] != i][:k])
        union = len(a | b)
        values.append(len(a & b) / union if union else np.nan)
    return float(np.nanmean(values))


def evaluate_trajectory_quality(
    scores,
    expression,
    embedding=None,
    window_fraction=0.1,
    n_neighbors=15,
):
    """Return a compact dictionary of manuscript trajectory quality metrics."""
    X = _as_2d(expression, "expression")
    metrics = {
        "smoothness_mean": float(np.nanmean(smoothness_score(scores, X))),
        "reconstruction_mse": trajectory_reconstruction_mse(
            scores,
            X,
            window_fraction=window_fraction,
        ),
        "mutual_information_mean": float(np.nanmean(pseudotime_mutual_information(scores, X))),
    }
    if embedding is not None:
        metrics["neighbor_jaccard"] = neighbor_jaccard(scores, embedding, n_neighbors=n_neighbors)
    return metrics


def _as_1d(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return arr


def _as_2d(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    return arr


def _check_same_rows(a, b):
    if len(a) != b.shape[0]:
        raise ValueError("scores and expression/embedding must have the same number of rows.")
