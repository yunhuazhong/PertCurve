import numpy as np

from pertcurve.metrics import (
    balanced_subsample_indices,
    evaluate_trajectory_quality,
    neighbor_jaccard,
    pseudotime_mutual_information,
    smoothness_score,
    trajectory_reconstruction_mse,
)


def test_balanced_subsample_indices_equalizes_groups():
    labels = np.array(["control"] * 20 + ["pert"] * 7)
    idx = balanced_subsample_indices(labels, random_state=1)

    sampled = labels[idx]
    assert (sampled == "control").sum() == 7
    assert (sampled == "pert").sum() == 7


def test_trajectory_quality_metrics_are_finite():
    scores = np.linspace(0, 1, 50)
    expr = np.column_stack([scores, scores**2])
    embedding = np.column_stack([scores, scores**2, np.sin(scores)])

    smooth = smoothness_score(scores, expr)
    recon = trajectory_reconstruction_mse(scores, expr, window_size=5)
    mi = pseudotime_mutual_information(scores, expr, n_neighbors=3)
    jac = neighbor_jaccard(scores, embedding, n_neighbors=5)
    summary = evaluate_trajectory_quality(scores, expr, embedding=embedding, n_neighbors=5)

    assert smooth.shape == (2,)
    assert np.all(np.isfinite(smooth))
    assert np.isfinite(recon)
    assert mi.shape == (2,)
    assert np.all(np.isfinite(mi))
    assert 0.0 <= jac <= 1.0
    assert {"smoothness_mean", "reconstruction_mse", "mutual_information_mean", "neighbor_jaccard"}.issubset(summary)
