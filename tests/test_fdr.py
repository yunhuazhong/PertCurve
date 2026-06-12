import numpy as np

from pertcurve.fdr import compute_all_fdr, trajectory_f_test


def test_trajectory_f_test_detects_linear_trend():
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 1, 80)
    expression = 2.0 * scores + rng.normal(scale=0.05, size=len(scores))

    result = trajectory_f_test(scores, expression, poly_order=1, min_cells=20)

    assert result["p_value"] < 1e-10
    assert result["r_squared"] > 0.9
    assert result["trend_direction"] == "rising"


def test_trajectory_f_test_constant_expression_is_flat():
    scores = np.linspace(0, 1, 80)
    expression = np.ones_like(scores)

    result = trajectory_f_test(scores, expression, poly_order=3, min_cells=20)

    assert result["p_value"] == 1.0
    assert result["r_squared"] == 0.0
    assert result["trend_direction"] == "flat"


def test_compute_all_fdr_adds_columns():
    scores = np.linspace(0, 1, 60)
    pert_data = {
        "PERT1": {"scores": scores, "gene_exprs": {"GENE1": scores, "GENE2": np.ones_like(scores)}},
        "PERT2": {"scores": scores, "gene_exprs": {"GENE1": 1 - scores, "GENE2": np.ones_like(scores)}},
    }

    df = compute_all_fdr(pert_data, ["GENE1", "GENE2"], poly_order=1, min_cells=20, n_jobs=1)

    assert {"fdr", "significant"}.issubset(df.columns)
    assert len(df) == 4
