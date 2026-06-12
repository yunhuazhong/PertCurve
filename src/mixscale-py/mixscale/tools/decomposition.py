"""Decomposition module for Mixscale - PCA-based permutation tests."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import chi2
from typing import Optional, Dict, Tuple, List, Union
import warnings


def pca_permtest(
    mat: Union[np.ndarray, pd.DataFrame],
    k: Optional[int] = 1,
    var_prop: Optional[float] = None,
    var_prop_total: Optional[float] = None,
    center: bool = True,
    scale: bool = True,
    row_filtering_pval: float = 0.05,
    num_iter: int = 200,
    seed: int = 123124125,
) -> Dict:
    """
    Run PCA-based permutation test for a Z-score matrix.

    Parameters
    ----------
    mat : np.ndarray or pd.DataFrame
        Z-score matrix (genes x conditions/samples)
    k : int, optional
        Number of top PCs to extract
    var_prop : float, optional
        Cutoff for proportion of variance explained
    var_prop_total : float, optional
        Cumulative variance explained cutoff
    center : bool
        Center columns (default: True)
    scale : bool
        Scale columns to unit variance (default: True)
    row_filtering_pval : float
        P-value threshold for row filtering (default: 0.05)
    num_iter : int
        Number of permutation iterations (default: 200)
    seed : int
        Random seed (default: 123124125)

    Returns
    -------
    dict
        Dictionary with 'mat', 'pmat', 'pca_obj'
    """
    np.random.seed(seed)

    if isinstance(mat, pd.DataFrame):
        row_names = mat.index.tolist()
        col_names = mat.columns.tolist()
        mat_array = mat.values
    else:
        row_names = None
        col_names = None
        mat_array = mat

    # Filter rows
    if row_filtering_pval > 0 and row_filtering_pval <= 1:
        chi2_threshold = np.sqrt(chi2.ppf(1 - row_filtering_pval, df=1))
        row_mask = ~np.all(np.abs(mat_array) < chi2_threshold, axis=1)
        n_removed = np.sum(~row_mask)

        print(f"Removing {n_removed} rows given row_filtering_pval = {row_filtering_pval}")

        if n_removed > 0:
            mat_array = mat_array[row_mask, :]
            if row_names is not None:
                row_names = [row_names[i] for i in range(len(row_names)) if row_mask[i]]

    # Perform PCA
    if center and scale:
        mat_scaled = (mat_array - np.mean(mat_array, axis=0)) / np.std(mat_array, axis=0)
    elif center:
        mat_scaled = mat_array - np.mean(mat_array, axis=0)
    elif scale:
        mat_scaled = mat_array / np.std(mat_array, axis=0)
    else:
        mat_scaled = mat_array

    pca = PCA()
    pca.fit(mat_scaled.T)
    test_scores = pca.transform(mat_scaled.T)

    # Get variance explained
    prop_test = pca.explained_variance_ratio_

    # Determine k
    if k is not None:
        pass
    elif var_prop is not None:
        k_idx = np.where(prop_test >= var_prop)[0]
        if len(k_idx) == 0:
            raise ValueError(
                f"None of the PC has %var >= {var_prop}. Please use a lower value."
            )
        k = k_idx[-1] + 1
    elif var_prop_total is not None:
        k = np.where(np.cumsum(prop_test) >= var_prop_total)[0][0] + 1
    else:
        raise ValueError(
            "Please provide at least one of: k, var_prop, var_prop_total."
        )

    # Permutation test
    n_genes = mat_scaled.shape[0]
    null_pc = np.zeros((n_genes * num_iter, k))

    for idx_iter in range(num_iter):
        # Permute each column independently
        null_tmp = np.apply_along_axis(np.random.permutation, 0, mat_scaled)

        # Perform PCA on permuted data
        pca_null = PCA(n_components=k)
        null_scores = pca_null.fit_transform(null_tmp.T)

        # Store results
        start_idx = idx_iter * n_genes
        end_idx = (idx_iter + 1) * n_genes
        null_pc[start_idx:end_idx, :] = null_scores

    # Calculate p-values
    pval = np.zeros((n_genes, k))

    for i in range(k):
        # Create ECDF
        sorted_null = np.sort(null_pc[:, i])

        for j in range(n_genes):
            # Find proportion of null values <= observed value
            pval[j, i] = np.searchsorted(sorted_null, test_scores[j, i]) / len(
                sorted_null
            )

    # Create output
    pval_df = pd.DataFrame(
        pval,
        columns=[f"PC{i+1}" for i in range(k)],
        index=row_names if row_names is not None else None,
    )

    mat_df = pd.DataFrame(
        mat_array,
        index=row_names if row_names is not None else None,
        columns=col_names if col_names is not None else None,
    )

    return {"mat": mat_df, "pmat": pval_df, "pca_obj": pca, "scores": test_scores}


def get_sig_genes(
    perm_obj: Dict,
    k: Optional[int] = 1,
    var_prop: Optional[float] = None,
    var_prop_total: Optional[float] = None,
    perm_pval_thres: float = 0.05,
    ori_pval_thres: float = 1.666667e-06,
    cor_threshold: float = 0.2,
    collapse: bool = True,
) -> Dict[str, List[str]]:
    """
    Extract significant genes from PCA permutation test.

    Parameters
    ----------
    perm_obj : dict
        Output from pca_permtest()
    k : int, optional
        Number of top PCs
    var_prop : float, optional
        Variance proportion cutoff
    var_prop_total : float, optional
        Cumulative variance cutoff
    perm_pval_thres : float
        Permutation p-value threshold (default: 0.05)
    ori_pval_thres : float
        Original DE test p-value threshold (default: 1.666667e-06)
    cor_threshold : float
        Correlation threshold for PC orientation (default: 0.2)
    collapse : bool
        Collapse genes from all PCs (default: True)

    Returns
    -------
    dict
        Dictionary with 'downDEGs' and 'upDEGs' lists
    """
    pca = perm_obj["pca_obj"]
    mat = perm_obj["mat"].values
    pval = perm_obj["pmat"].values
    scores = perm_obj["scores"]

    # Get variance explained
    prop_test = pca.explained_variance_ratio_

    # Determine k
    if k is not None:
        max_pc_idx = k
    elif var_prop is not None:
        idx1 = np.where(prop_test >= var_prop)[0]
        max_pc_idx = idx1[-1] + 1
    elif var_prop_total is not None:
        max_pc_idx = np.where(np.cumsum(prop_test) >= var_prop_total)[0][0] + 1
    else:
        raise ValueError(
            "Please provide at least one of: k, var_prop, var_prop_total."
        )

    # Select p-values for top PCs
    slct_pval = pval[:, :max_pc_idx]

    # Calculate Z-score threshold
    z_threshold = np.sqrt(chi2.ppf(1 - ori_pval_thres, df=1))

    # Extract significant genes for each PC
    top_deg_idx = {}
    bottom_deg_idx = {}

    for i in range(max_pc_idx):
        # Calculate correlation between PC and columns
        cor_test = np.corrcoef(scores[:, i], mat.T)[0, 1:]

        # Find columns significantly correlated with PC
        prtb_idx = np.where(np.abs(cor_test) >= cor_threshold)[0]

        # Find top and bottom DEGs
        top_mask = slct_pval[:, i] >= (1 - perm_pval_thres)
        if len(prtb_idx) > 0:
            ori_sig_mask = np.any(np.abs(mat[:, prtb_idx]) >= z_threshold, axis=1)
        else:
            ori_sig_mask = np.ones(mat.shape[0], dtype=bool)

        top_deg = np.where(top_mask & ori_sig_mask)[0]
        top_deg = top_deg[np.argsort(scores[top_deg, i])[::-1]]

        bottom_mask = slct_pval[:, i] <= perm_pval_thres
        bottom_deg = np.where(bottom_mask & ori_sig_mask)[0]
        bottom_deg = bottom_deg[np.argsort(scores[bottom_deg, i])[::-1]]

        # Determine orientation by sum of Z-scores
        if len(prtb_idx) > 0:
            top_sum = np.sum(mat[top_deg, :][:, prtb_idx])
            bottom_sum = np.sum(mat[bottom_deg, :][:, prtb_idx])

            if top_sum > bottom_sum:
                # Swap
                top_deg, bottom_deg = bottom_deg[::-1], top_deg[::-1]

        top_deg_idx[f"PC{i+1}"] = top_deg
        bottom_deg_idx[f"PC{i+1}"] = bottom_deg

    # Get gene names if available
    if perm_obj["mat"].index is not None:
        gene_names = perm_obj["mat"].index.tolist()
    else:
        gene_names = [f"gene_{i}" for i in range(mat.shape[0])]

    # Collapse or return separately
    if collapse:
        # Combine all PCs
        top_all = []
        for pc in top_deg_idx.values():
            top_all.extend(pc)
        top_all = list(dict.fromkeys(top_all))  # Remove duplicates, preserve order

        bottom_all = []
        for pc_name in reversed(list(bottom_deg_idx.keys())):
            bottom_all.extend(bottom_deg_idx[pc_name][::-1])
        bottom_all = list(dict.fromkeys(bottom_all))

        return {
            "downDEGs": [gene_names[i] for i in top_all],
            "upDEGs": [gene_names[i] for i in bottom_all],
        }
    else:
        down_degs = {}
        up_degs = {}

        for pc_name in top_deg_idx.keys():
            down_degs[pc_name] = [gene_names[i] for i in top_deg_idx[pc_name]]
            up_degs[pc_name] = [
                gene_names[i] for i in bottom_deg_idx[pc_name][::-1]
            ]

        return {"downDEGs": down_degs, "upDEGs": up_degs}
