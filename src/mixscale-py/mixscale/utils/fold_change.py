"""Utility functions for fold change calculations."""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union, Tuple


def get_fold_change(
    gene_exp: np.ndarray,
    idx_p: np.ndarray,
    idx_nt: np.ndarray,
    min_cells: int = 3,
    thresh_min: float = 0.0,
    pseudocount_use: float = 1.0,
    min_pct: float = 0.1,
    base: int = 2,
    norm_method: str = "raw",
) -> float:
    """
    Calculate log-fold-change for a gene.

    Parameters
    ----------
    gene_exp : np.ndarray
        Vector of gene expression levels
    idx_p : np.ndarray
        Indices of perturbed cells in gene_exp
    idx_nt : np.ndarray
        Indices of non-target cells (controls) in gene_exp
    min_cells : int
        Minimal number of cells that express the gene (default: 3)
    thresh_min : float
        Minimal expression value; values below are considered 0 (default: 0.0)
    pseudocount_use : float
        Small value added to log-transformation to avoid log(0) (default: 1.0)
    min_pct : float
        Minimal proportion of cells expressing the gene (default: 0.1)
    base : int
        Base for logarithm (default: 2)
    norm_method : str
        Normalization method: 'raw', 'log_norm', or 'scale_data' (default: 'raw')

    Returns
    -------
    float
        Log-fold-change value, or np.nan if criteria not met
    """
    # Flag 1: minimum cell check
    if (
        np.sum(gene_exp[idx_p] > 0) < min_cells
        and np.sum(gene_exp[idx_nt] > 0) < min_cells
    ):
        return np.nan

    # Flag 2: variance check (not 0)
    if np.var(gene_exp) == 0:
        return np.nan

    # Flag 3: min_pct check
    pct_1 = np.round(np.sum(gene_exp[idx_nt] > thresh_min) / len(idx_nt), 3)
    pct_2 = np.round(np.sum(gene_exp[idx_p] > thresh_min) / len(idx_p), 3)

    if pct_1 < min_pct and pct_2 < min_pct:
        return np.nan

    # Set the mean function according to norm_method
    def default_mean_fn(x):
        return np.log(np.mean(x) + pseudocount_use) / np.log(base)

    if norm_method == "log_norm":
        mean_fn = lambda x: np.log(np.mean(np.expm1(x)) + pseudocount_use) / np.log(base)
    elif norm_method == "scale_data":
        mean_fn = np.mean
    else:
        mean_fn = default_mean_fn

    # Calculate fold change
    data_1 = mean_fn(gene_exp[idx_nt])
    data_2 = mean_fn(gene_exp[idx_p])
    fc = data_2 - data_1

    return fc


def calculate_percent_expressed(
    data: Union[np.ndarray, sparse.spmatrix],
    cells: np.ndarray,
    thresh_min: float = 0.0,
) -> np.ndarray:
    """
    Calculate percentage of cells expressing each gene.

    Parameters
    ----------
    data : np.ndarray or sparse matrix
        Gene expression matrix (genes x cells)
    cells : np.ndarray
        Indices of cells to calculate percentage for
    thresh_min : float
        Minimum expression threshold (default: 0.0)

    Returns
    -------
    np.ndarray
        Percentage of cells expressing each gene
    """
    if sparse.issparse(data):
        data_subset = data[:, cells]
        pct = np.array((data_subset > thresh_min).sum(axis=1) / len(cells)).flatten()
    else:
        data_subset = data[:, cells]
        pct = np.sum(data_subset > thresh_min, axis=1) / len(cells)

    return np.round(pct, 3)


def fold_change_matrix(
    data: Union[np.ndarray, sparse.spmatrix],
    cells_1: np.ndarray,
    cells_2: np.ndarray,
    features: Optional[np.ndarray] = None,
    pseudocount_use: float = 1.0,
    base: int = 2,
    mean_fn: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate fold change for multiple genes.

    Parameters
    ----------
    data : np.ndarray or sparse matrix
        Gene expression matrix (genes x cells)
    cells_1 : np.ndarray
        Indices of first group of cells
    cells_2 : np.ndarray
        Indices of second group of cells
    features : np.ndarray, optional
        Indices of features (genes) to calculate for
    pseudocount_use : float
        Pseudocount for log transformation (default: 1.0)
    base : int
        Base for logarithm (default: 2)
    mean_fn : callable, optional
        Custom mean function

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (avg_group1, avg_group2, log_fc)
    """
    if features is None:
        features = np.arange(data.shape[0])

    # Default mean function
    if mean_fn is None:

        def mean_fn(x):
            return np.log(np.mean(x) + pseudocount_use) / np.log(base)

    # Calculate averages
    if sparse.issparse(data):
        avg_1 = np.array(
            [mean_fn(data[i, cells_1].toarray().flatten()) for i in features]
        )
        avg_2 = np.array(
            [mean_fn(data[i, cells_2].toarray().flatten()) for i in features]
        )
    else:
        avg_1 = np.array([mean_fn(data[i, cells_1]) for i in features])
        avg_2 = np.array([mean_fn(data[i, cells_2]) for i in features])

    log_fc = avg_2 - avg_1

    return avg_1, avg_2, log_fc
