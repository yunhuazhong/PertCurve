import numpy as np

from pertcurve import (
    principal_curve_binned,
    project_points_to_curve_continuous,
    score_perturbation,
)


def synthetic_data(seed=0):
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(loc=[0.0, 0.0], scale=0.15, size=(80, 2))
    pert = rng.normal(loc=[2.0, 0.8], scale=0.20, size=(90, 2))
    X = np.vstack([ctrl, pert])
    labels = np.array(["control"] * len(ctrl) + ["pert"] * len(pert))
    return X, labels


def test_principal_curve_and_projection_shapes():
    X, labels = synthetic_data()
    curve, bin_centers = principal_curve_binned(X, labels, n_bins=8, smoothing=0.2)
    projected, arc_lengths, distances = project_points_to_curve_continuous(X, curve)

    assert curve.shape == (100, 2)
    assert bin_centers.shape[1] == 2
    assert projected.shape == X.shape
    assert arc_lengths.shape == (X.shape[0],)
    assert np.all(distances >= 0)


def test_score_centroids_are_near_zero_and_one():
    X, labels = synthetic_data()
    df_scores, stats, _, _ = score_perturbation(X, labels, labels, n_bins=8, smoothing=0.2)
    ctrl_mean = df_scores.loc[df_scores["perturbation"] == "control", "normalized_pseudotime"].mean()
    pert_mean = df_scores.loc[df_scores["perturbation"] == "pert", "normalized_pseudotime"].mean()

    assert abs(ctrl_mean) < 0.25
    assert 0.75 < pert_mean < 1.25
    assert stats["n_cells_ctrl"] == 80
    assert stats["n_cells_pert"] == 90


def test_near_overlapping_centroids_do_not_crash():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3)) * 0.01
    labels = np.array(["control"] * 20 + ["pert"] * 20)
    df_scores, stats, curve, _ = score_perturbation(X, labels, labels, n_bins=4)

    assert len(df_scores) == 40
    assert curve.shape == (100, 3)
    assert np.isfinite(df_scores["normalized_pseudotime"]).all()
    assert np.isfinite(stats["curve_length"])
