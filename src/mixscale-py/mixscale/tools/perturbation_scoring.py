"""Perturbation scoring module for Mixscale."""

import numpy as np
import pandas as pd
from anndata import AnnData
from typing import Optional, List, Dict, Union
import scanpy as sc
from scipy import sparse
import warnings
import tqdm

from ..utils.fold_change import get_fold_change, calculate_percent_expressed


def _top_de_genes(
    adata: AnnData,
    cells_1: np.ndarray,
    cells_2: np.ndarray,
    layer: str = "counts",
    logfc_threshold: float = 0.25,
    pval_cutoff: float = 0.05,
    min_pct: float = 0.1,
) -> List[str]:
    """
    Find top DE genes between two groups of cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    cells_1 : np.ndarray
        Indices of first group of cells
    cells_2 : np.ndarray
        Indices of second group of cells
    layer : str
        Layer to use for DE test (default: 'counts')
    logfc_threshold : float
        Log-fold-change threshold (default: 0.25)
    pval_cutoff : float
        P-value cutoff (default: 0.05)
    min_pct : float
        Minimum fraction of cells expressing (default: 0.1)

    Returns
    -------
    List[str]
        List of top DE gene names
    """
    # Create a temporary subset for DE testing
    subset_idx = np.concatenate([cells_1, cells_2])
    adata_sub = adata[subset_idx, :].copy()

    # Create group labels
    groups = np.array(["group2"] * len(subset_idx))
    groups[: len(cells_1)] = "group1"
    adata_sub.obs["_de_group"] = groups

    if layer == "counts":
        layer = None

    # Run rank genes test
    sc.tl.rank_genes_groups(
        adata_sub,
        groupby="_de_group",
        groups=["group1"],
        reference="group2",
        method="wilcoxon",
        layer=layer,
    )

    # Extract results
    result = sc.get.rank_genes_groups_df(adata_sub, group="group1")

    # Filter by logFC and pvalue
    result = result[
        (np.abs(result["logfoldchanges"]) > logfc_threshold)
        & (result["pvals_adj"] < pval_cutoff)
    ]

    # Sort by absolute log fold change
    result = result.sort_values("logfoldchanges", key=abs, ascending=False)

    return result["names"].tolist()


def run_mixscale(
    adata: AnnData,
    labels: str = "gene",
    nt_class_name: str = "NT",
    new_class_name: str = "mixscale_score",
    min_de_genes: int = 5,
    min_cells: int = 5,
    layer: str = "counts",
    logfc_threshold: float = 0.25,
    verbose: bool = False,
    split_by: Optional[str] = None,
    fine_mode: bool = False,
    fine_mode_labels: str = "guide_ID",
    de_gene: Optional[Dict[str, List[str]]] = None,
    max_de_genes: int = 100,
    harmonize: bool = False,
    min_prop_ntgd: float = 0.1,
    pval_cutoff: float = 0.05,
    seed: int = 10282021,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Calculate Mixscale perturbation scores.

    This function calculates perturbation scores for perturbed and non-perturbed
    gRNA expressing cells. The perturbation score reflects the perturbation strength
    of each cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    labels : str
        Column in adata.obs with target gene labels (default: 'gene')
    nt_class_name : str
        Classification name of non-targeting gRNA cells (default: 'NT')
    new_class_name : str
        Name of mixscale scores to store in adata.obs (default: 'mixscale_score')
    min_de_genes : int
        Required number of DE genes (default: 5)
    min_cells : int
        Minimum number of cells in target gene class (default: 5)
    layer : str
        Layer to use for expression data (default: 'counts')
    logfc_threshold : float
        Log-fold-change threshold for DE genes (default: 0.25)
    verbose : bool
        Display messages (default: False)
    split_by : str, optional
        Column in adata.obs with cell type/condition labels
    fine_mode : bool
        Calculate DE genes per gRNA separately (default: False)
    fine_mode_labels : str
        Column in adata.obs with gRNA ID labels (default: 'guide_ID')
    de_gene : dict, optional
        User-defined DE genes for each perturbation
    max_de_genes : int
        Maximum number of top DE genes to use (default: 100)
    harmonize : bool
        Harmonize cell-type proportion between NT and perturbed (default: False)
    min_prop_ntgd : float
        Minimal threshold for cell type proportion (default: 0.1)
    pval_cutoff : float
        DE test p-value cutoff (default: 0.05)
    seed : int
        Random seed (default: 10282021)
    copy : bool
        Return a copy of adata (default: False)

    Returns
    -------
    AnnData or None
        Updated AnnData object if copy=True, otherwise modifies in place.
        Adds two columns to adata.obs:
        - {new_class_name}: Standardized perturbation scores
        - {new_class_name}_class: Classification labels (gene name or "gene NP" for 
          non-perturbed cells with insufficient DE genes)
        Also stores detailed results in adata.uns['mixscale_scores']
    """
    if verbose:
        print("Calculating Mixscale scores...")

    np.random.seed(seed)

    if copy:
        adata = adata.copy()

    # Check if labels column exists
    if labels not in adata.obs.columns:
        raise ValueError(f"Column '{labels}' not found in adata.obs")

    # Get expression data
    if layer == "counts":
        if sparse.issparse(adata.X):
            data = adata.X.toarray().T  # Transpose to genes x cells
        else:
            data = adata.X.T
    else:
        if sparse.issparse(adata.layers[layer]):
            data = adata.layers[layer].toarray().T
        else:
            data = adata.layers[layer].T

    # Initialize storage
    prtb_markers = {}
    gv_list = {}

    # Handle split_by
    if split_by is None:
        splits = ["con1"]
        adata.obs["_split"] = "con1"
        split_by = "_split"
    else:
        splits = adata.obs[split_by].unique().tolist()

    # Initialize new columns
    adata.obs[new_class_name] = 0.0
    adata.obs[f"{new_class_name}_class"] = adata.obs[labels].astype(str)

    # Get all genes except NT
    all_genes = [g for g in adata.obs[labels].unique() if g != nt_class_name]

    # Process each perturbation
    for gene in tqdm.tqdm(all_genes, desc="Processing perturbations"):
        if verbose:
            print(f"Processing {gene}...")

        gv_list[gene] = {}

        for s in splits:
            # Get cells for this split
            split_mask = adata.obs[split_by] == s
            guide_cells_mask = (adata.obs[labels] == gene) & split_mask
            nt_cells_mask = (adata.obs[labels] == nt_class_name) & split_mask

            guide_cells = np.where(guide_cells_mask)[0]
            nt_cells = np.where(nt_cells_mask)[0]

            if len(guide_cells) < min_cells:
                if verbose:
                    print(f"  Too few cells for {gene} in {s}, skipping...")
                continue

            # Get DE genes
            if de_gene is not None and gene in de_gene:
                all_de_genes = de_gene[gene]
            else:
                all_de_genes = _top_de_genes(
                    adata,
                    guide_cells,
                    nt_cells,
                    layer=layer,
                    logfc_threshold=logfc_threshold,
                    pval_cutoff=pval_cutoff,
                )

            # Limit to max_de_genes
            if len(all_de_genes) > max_de_genes:
                if verbose:
                    print(
                        f"  Restricting to top {max_de_genes} genes (from {len(all_de_genes)})..."
                    )
                all_de_genes = all_de_genes[:max_de_genes]

            if len(all_de_genes) < min_de_genes:
                if verbose:
                    print(f"  Fewer than {min_de_genes} DE genes for {gene}, skipping...")
                # Assign cells as NP (non-perturbed)
                adata.obs.loc[guide_cells_mask, f"{new_class_name}_class"] = f"{gene} NP"
                continue

            # Get expression data for DE genes
            de_gene_idx = [i for i, g in enumerate(adata.var_names) if g in all_de_genes]
            all_cells = np.concatenate([guide_cells, nt_cells])

            dat = data[de_gene_idx, :][:, all_cells]

            # Calculate perturbation score
            guide_mean = np.mean(dat[:, : len(guide_cells)], axis=1)
            nt_mean = np.mean(dat[:, len(guide_cells) :], axis=1)
            vec = guide_mean - nt_mean

            # Calculate scores for all cells
            pvec_mat = (dat.T * vec).T  # genes x cells
            vec_mat = vec * vec
            pvec = np.sum(pvec_mat, axis=0) / np.sum(vec_mat)

            # Store results
            gv = pd.DataFrame({"pvec": pvec, "gene": nt_class_name}, index=all_cells)
            gv.loc[guide_cells, "gene"] = gene

            # Collect columns to add in a dictionary
            columns_to_add = {}
            for omit_idx in range(len(de_gene_idx)):
                remain_idx = [i for i in range(len(de_gene_idx)) if i != omit_idx]
                pvec_loo = np.sum(pvec_mat[remain_idx, :], axis=0) / np.sum(
                    vec_mat[remain_idx]
                )
                columns_to_add[all_de_genes[omit_idx]] = pvec_loo

            # Add all columns at once using pd.concat
            new_columns_df = pd.DataFrame(columns_to_add, index=gv.index)
            gv = pd.concat([gv, new_columns_df], axis=1)

            gv_list[gene][s] = gv

    # Calculate standardized scores
    all_score = []

    # breakpoint()
    for prtb in all_genes:
        if prtb not in gv_list or len(gv_list[prtb]) == 0:
            # No scores calculated, use binary 0/1
            prtb_mask = adata.obs[labels] == prtb
            nt_mask = adata.obs[labels] == nt_class_name
            mask = prtb_mask | nt_mask

            masked_indices = np.where(mask)[0]
            prtb_in_masked = np.where(prtb_mask & mask)[0]
            
            scores = pd.DataFrame(
                {"cell_idx": masked_indices, "weight": 0.0}
            )
            # Set weight to 1.0 for perturbed cells using boolean indexing
            scores.loc[scores["cell_idx"].isin(prtb_in_masked), "weight"] = 1.0
            all_score.append(scores)
        else:
            # Use calculated scores
            for celltype in gv_list[prtb].keys():
                tmp = gv_list[prtb][celltype]
                idx_nt = tmp["gene"] == nt_class_name
                idx_gene = tmp["gene"] == prtb

                # Standardize scores
                mean_nt = tmp.loc[idx_nt, "pvec"].mean()
                sd_nt = tmp.loc[idx_nt, "pvec"].std()

                std_weight = (tmp.loc[idx_gene, "pvec"] - mean_nt) / sd_nt

                weights = pd.Series(0.0, index=tmp.index)
                weights.loc[idx_gene] = std_weight

                scores = pd.DataFrame({"cell_idx": tmp.index, "weight": weights.values})
                all_score.append(scores)

    # Combine all scores
    if all_score:
        all_score_df = pd.concat(all_score, ignore_index=True)
        all_score_df = all_score_df.drop_duplicates(subset="cell_idx")
        all_score_df = all_score_df.sort_values("cell_idx")

        # Add to adata.obs
        adata.obs[new_class_name] = 0.0
        adata.obs.iloc[all_score_df["cell_idx"].values, adata.obs.columns.get_loc(new_class_name)] = all_score_df["weight"].values

    # Store detailed results in uns
    adata.uns["mixscale_scores"] = gv_list

    if verbose:
        print("Done calculating Mixscale scores.")

    if copy:
        return adata
