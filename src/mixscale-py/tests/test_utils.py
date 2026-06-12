"""Tests for Mixscale utility functions."""

import numpy as np
import pytest
from mixscale.utils import get_fold_change, calculate_percent_expressed


def test_get_fold_change():
    """Test fold change calculation."""
    # Create simple test data
    gene_exp = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    idx_p = np.array([5, 6, 7, 8, 9])  # Perturbed cells
    idx_nt = np.array([0, 1, 2, 3, 4])  # NT cells

    fc = get_fold_change(
        gene_exp=gene_exp,
        idx_p=idx_p,
        idx_nt=idx_nt,
        min_cells=3,
        pseudocount_use=1.0,
        min_pct=0.1,
        base=2,
    )

    # Should return a positive fold change
    assert fc > 0, "Fold change should be positive for upregulated gene"


def test_get_fold_change_no_variance():
    """Test fold change with zero variance."""
    gene_exp = np.array([5, 5, 5, 5, 5])
    idx_p = np.array([0, 1])
    idx_nt = np.array([2, 3, 4])

    fc = get_fold_change(
        gene_exp=gene_exp, idx_p=idx_p, idx_nt=idx_nt, min_cells=1
    )

    # Should return NaN for zero variance
    assert np.isnan(fc), "Should return NaN for zero variance"


def test_calculate_percent_expressed():
    """Test percent expressed calculation."""
    # Create simple expression matrix (genes x cells)
    data = np.array([[0, 0, 1, 2], [0, 1, 2, 3], [0, 0, 0, 0]])

    cells = np.array([0, 1, 2, 3])
    pct = calculate_percent_expressed(data, cells)

    expected = np.array([0.5, 0.75, 0.0])
    np.testing.assert_array_almost_equal(pct, expected, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
