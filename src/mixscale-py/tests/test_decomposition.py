"""Tests for decomposition functions."""

import numpy as np
import pandas as pd
import pytest
from mixscale.tools import pca_permtest, get_sig_genes


def test_pca_permtest():
    """Test PCA permutation test."""
    # Create simple Z-score matrix
    np.random.seed(42)
    mat = pd.DataFrame(
        np.random.randn(100, 10),  # 100 genes x 10 conditions
        index=[f"gene_{i}" for i in range(100)],
        columns=[f"cond_{i}" for i in range(10)],
    )

    # Run permutation test
    result = pca_permtest(mat, k=3, num_iter=50, seed=42)

    # Check outputs
    assert "mat" in result
    assert "pmat" in result
    assert "pca_obj" in result
    assert result["pmat"].shape == (100, 3)


def test_get_sig_genes():
    """Test significant gene extraction."""
    # Create test data
    np.random.seed(42)
    mat = pd.DataFrame(
        np.random.randn(100, 10),
        index=[f"gene_{i}" for i in range(100)],
        columns=[f"cond_{i}" for i in range(10)],
    )

    # Add some strong signals
    mat.iloc[0:5, :] = 5  # Strong upregulation
    mat.iloc[95:100, :] = -5  # Strong downregulation

    perm_obj = pca_permtest(mat, k=2, num_iter=50, seed=42)

    sig_genes = get_sig_genes(
        perm_obj, k=2, perm_pval_thres=0.05, ori_pval_thres=0.05, collapse=True
    )

    # Check outputs
    assert "downDEGs" in sig_genes
    assert "upDEGs" in sig_genes
    assert isinstance(sig_genes["downDEGs"], list)
    assert isinstance(sig_genes["upDEGs"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
