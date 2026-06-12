"""Utility functions for GLM fitting with Gamma-Poisson distribution."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import digamma, polygamma
from scipy.optimize import minimize
from typing import Optional, Union, Tuple, Dict
import warnings


def estimate_size_factors(
    counts: Union[np.ndarray, sparse.spmatrix],
    method: str = "normed_sum",
) -> np.ndarray:
    """
    Estimate size factors for normalization.

    Parameters
    ----------
    counts : np.ndarray or sparse matrix
        Count matrix (genes x cells)
    method : str
        Method for size factor estimation: 'normed_sum', 'median_ratio' (default: 'normed_sum')

    Returns
    -------
    np.ndarray
        Size factors for each cell
    """
    if method == "normed_sum":
        if sparse.issparse(counts):
            lib_sizes = np.array(counts.sum(axis=0)).flatten()
        else:
            lib_sizes = counts.sum(axis=0)
        size_factors = lib_sizes / np.mean(lib_sizes)

    elif method == "median_ratio":
        # Geometric mean of each gene across cells
        if sparse.issparse(counts):
            counts_dense = counts.toarray()
        else:
            counts_dense = counts

        # Avoid zeros in geometric mean calculation
        pseudo_refs = np.exp(
            np.mean(np.log(counts_dense + 1), axis=1, keepdims=True)
        ) - 1
        pseudo_refs[pseudo_refs == 0] = 1

        # Calculate size factors as median of ratios
        ratios = counts_dense / pseudo_refs
        size_factors = np.median(ratios, axis=0)
        size_factors[size_factors == 0] = 1

    else:
        raise ValueError(f"Unknown method: {method}")

    return size_factors


def nb_nll(
    theta: float,
    mu: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Negative binomial negative log-likelihood.

    Parameters
    ----------
    theta : float
        Dispersion parameter (1/overdispersion)
    mu : np.ndarray
        Mean parameters
    y : np.ndarray
        Observed counts

    Returns
    -------
    float
        Negative log-likelihood
    """
    if theta <= 0:
        return np.inf

    # Negative binomial log-likelihood
    nll = -np.sum(
        y * np.log(mu / (mu + theta))
        + theta * np.log(theta / (mu + theta))
        + np.sum([np.log(1 + i / theta) for i in range(int(np.max(y)))])
    )

    return nll


def estimate_dispersion_mle(
    counts: np.ndarray,
    mu: np.ndarray,
    min_disp: float = 1e-8,
    max_disp: float = 100,
) -> np.ndarray:
    """
    Estimate overdispersion parameters using MLE.

    Parameters
    ----------
    counts : np.ndarray
        Observed count data (genes x cells)
    mu : np.ndarray
        Fitted means (genes x cells)
    min_disp : float
        Minimum dispersion value (default: 1e-8)
    max_disp : float
        Maximum dispersion value (default: 100)

    Returns
    -------
    np.ndarray
        Estimated dispersions for each gene
    """
    n_genes = counts.shape[0]
    dispersions = np.zeros(n_genes)

    for i in range(n_genes):
        y = counts[i, :]
        m = mu[i, :]

        # Skip if all zeros
        if np.sum(y) == 0:
            dispersions[i] = min_disp
            continue

        # Initial guess
        theta_init = 1.0

        # Optimize
        try:
            result = minimize(
                lambda theta: nb_nll(theta, m, y),
                theta_init,
                bounds=[(1e-8, 1e4)],
                method="L-BFGS-B",
            )
            theta = result.x[0]
            # Convert theta to overdispersion (alpha)
            alpha = 1.0 / theta
            dispersions[i] = np.clip(alpha, min_disp, max_disp)
        except:
            dispersions[i] = 0.1  # Default value

    return dispersions


def fit_glm_gp(
    counts: Union[np.ndarray, sparse.spmatrix],
    design_matrix: np.ndarray,
    size_factors: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    estimate_overdispersion: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Fit a Gamma-Poisson GLM.

    Parameters
    ----------
    counts : np.ndarray or sparse matrix
        Count matrix (genes x cells)
    design_matrix : np.ndarray
        Design matrix (cells x covariates)
    size_factors : np.ndarray, optional
        Size factors for each cell
    offset : np.ndarray, optional
        Offset matrix
    estimate_overdispersion : bool
        Whether to estimate overdispersion (default: True)
    verbose : bool
        Print progress messages (default: False)

    Returns
    -------
    dict
        Dictionary with 'beta', 'overdispersion', 'mu', 'size_factors'
    """
    if sparse.issparse(counts):
        counts = counts.toarray()

    n_genes, n_cells = counts.shape

    # Estimate size factors if not provided
    if size_factors is None:
        if verbose:
            print("Estimating size factors...")
        size_factors = estimate_size_factors(counts)

    # Create offset matrix
    if offset is None:
        offset = np.log(size_factors)[np.newaxis, :]
    else:
        offset = offset + np.log(size_factors)[np.newaxis, :]

    # Initialize beta coefficients
    if verbose:
        print("Estimating beta coefficients...")

    n_coef = design_matrix.shape[1]
    beta = np.zeros((n_genes, n_coef))

    # Fit GLM for each gene using simple Poisson regression
    for i in range(n_genes):
        y = counts[i, :]

        # Skip if all zeros
        if np.sum(y) == 0:
            continue

        # Use log-linear model: log(mu) = X * beta + offset
        # Initial estimate using log-transformed data
        log_y = np.log(y + 1)
        try:
            beta[i, :] = np.linalg.lstsq(design_matrix, log_y - offset[0, :], rcond=None)[0]
        except:
            beta[i, :] = 0

    # Calculate mu
    mu = np.exp(design_matrix @ beta.T + offset.T).T

    # Estimate overdispersion
    if estimate_overdispersion:
        if verbose:
            print("Estimating overdispersion...")
        overdispersion = estimate_dispersion_mle(counts, mu)
    else:
        overdispersion = np.zeros(n_genes)

    return {
        "beta": beta,
        "overdispersion": overdispersion,
        "mu": mu,
        "size_factors": size_factors,
    }
