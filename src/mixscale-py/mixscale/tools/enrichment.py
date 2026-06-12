"""Enrichment testing module for Mixscale."""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom
from typing import List, Dict, Optional, Union
import warnings


def fisher_enrich_test(
    input_list: List[str],
    background: List[str],
    go_term_db: Dict[str, List[str]],
    min_size: int = 5,
    max_size: int = 500,
) -> pd.DataFrame:
    """
    Fisher's exact test for gene set enrichment.

    Parameters
    ----------
    input_list : list
        List of genes to test
    background : list
        Background gene list
    go_term_db : dict
        Dictionary of gene sets {term_name: [gene1, gene2, ...]}
    min_size : int
        Minimum gene set size (default: 5)
    max_size : int
        Maximum gene set size (default: 500)

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns: term, overlap, pval, odds_ratio
    """
    results = []

    input_set = set(input_list)
    background_set = set(background)

    n_input = len(input_set & background_set)
    n_background = len(background_set)

    for term, genes in go_term_db.items():
        gene_set = set(genes) & background_set

        # Filter by size
        if len(gene_set) < min_size or len(gene_set) > max_size:
            continue

        # Calculate overlap
        overlap = input_set & gene_set
        n_overlap = len(overlap)

        if n_overlap == 0:
            continue

        # Fisher's exact test
        # Contingency table:
        #                In gene set    Not in gene set
        # In input           a                 b
        # Not in input       c                 d

        a = n_overlap
        b = n_input - a
        c = len(gene_set) - a
        d = n_background - n_input - c

        odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")

        results.append(
            {
                "term": term,
                "overlap": n_overlap,
                "gene_set_size": len(gene_set),
                "input_size": n_input,
                "overlap_genes": ",".join(sorted(overlap)),
                "pval": pval,
                "odds_ratio": odds_ratio,
            }
        )

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("pval")

    # Add adjusted p-values (Bonferroni)
    results_df["pval_adj"] = np.minimum(results_df["pval"] * len(results_df), 1.0)

    return results_df


def rbo(
    list1: np.ndarray,
    list2: np.ndarray,
    p: float = 0.9,
    k: Optional[int] = None,
) -> float:
    """
    Rank-biased overlap (RBO) calculation.

    Parameters
    ----------
    list1 : np.ndarray
        First ranked list
    list2 : np.ndarray
        Second ranked list
    p : float
        Weighting parameter [0, 1] (default: 0.9)
    k : int, optional
        Evaluation depth

    Returns
    -------
    float
        RBO score
    """
    if k is None:
        k = max(len(list1), len(list2)) // 2

    # Convert to sets for overlap calculation
    set1 = set(list1)
    set2 = set(list2)

    # Calculate agreement at each depth
    overlap_size = np.zeros(k)
    for d in range(1, k + 1):
        s1 = set(list1[: min(d, len(list1))])
        s2 = set(list2[: min(d, len(list2))])
        overlap_size[d - 1] = len(s1 & s2)

    # Calculate RBO
    agreement = overlap_size / np.arange(1, k + 1)

    # Weighted sum
    weights = np.array([p ** (d - 1) for d in range(1, k + 1)])
    rbo_score = (1 - p) * np.sum(weights * agreement)

    return min(1.0, rbo_score)


def de_enrich(
    adata,
    plist: Dict[str, List[str]],
    ident_1: str,
    ident_2: str,
    ident: str = None,
    split_by: Optional[str] = None,
    slct_ct: Optional[List[str]] = None,
    direction: str = "both",
    logfc_threshold: float = 0.25,
    p_val_cutoff: float = 0.05,
    min_pct: float = 0.1,
    layer: str = "counts",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Wrapper for DE test and enrichment test.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    plist : dict
        Pathway gene lists
    ident_1 : str
        First identity class
    ident_2 : str
        Second identity class
    ident : str, optional
        Column in adata.obs for grouping
    split_by : str, optional
        Column for cell type/condition
    slct_ct : list, optional
        Selected cell types
    direction : str
        Direction of DE: 'up', 'down', 'both' (default: 'both')
    logfc_threshold : float
        Log-fold-change threshold (default: 0.25)
    p_val_cutoff : float
        P-value cutoff (default: 0.05)
    min_pct : float
        Minimum percent cells (default: 0.1)
    layer : str
        Layer to use (default: 'counts')

    Returns
    -------
    pd.DataFrame or dict
        Enrichment results
    """
    import scanpy as sc

    if split_by is None:
        celltype_list = ["con1"]
        adata.obs["_split"] = "con1"
        split_by = "_split"
    else:
        celltype_list = sorted(adata.obs[split_by].unique())

    if slct_ct is not None:
        celltype_list = [ct for ct in celltype_list if ct in slct_ct]

    enrich_list = {}

    for celltype in celltype_list:
        # Subset to celltype
        mask = adata.obs[split_by] == celltype
        adata_sub = adata[mask, :].copy()

        if ident is not None:
            group_col = ident
        else:
            group_col = split_by

        # Create combined identity
        adata_sub.obs["_new_ident"] = (
            adata_sub.obs[split_by].astype(str)
            + "_"
            + adata_sub.obs[group_col].astype(str)
        )

        ident_1_tmp = f"{celltype}_{ident_1}"
        ident_2_tmp = f"{celltype}_{ident_2}"

        # Run DE test
        sc.tl.rank_genes_groups(
            adata_sub,
            groupby="_new_ident",
            groups=[ident_1_tmp],
            reference=ident_2_tmp,
            method="wilcoxon",
            layer=layer,
        )

        de_res = sc.get.rank_genes_groups_df(adata_sub, group=ident_1_tmp)

        # Get DEGs
        up_deg = de_res[
            (de_res["pvals_adj"] <= p_val_cutoff)
            & (de_res["logfoldchanges"] > logfc_threshold)
        ]["names"].tolist()

        down_deg = de_res[
            (de_res["pvals_adj"] <= p_val_cutoff)
            & (de_res["logfoldchanges"] < -logfc_threshold)
        ]["names"].tolist()

        background = de_res["names"].tolist()

        # Run enrichment
        enrich_res_down = None
        enrich_res_up = None

        if direction in ["down", "both"] and len(down_deg) >= 5:
            enrich_res_down = fisher_enrich_test(down_deg, background, plist)
            if len(enrich_res_down) > 0:
                enrich_res_down["num_DEG"] = len(down_deg)
                enrich_res_down["direction_DEG"] = "downDEG"
                enrich_res_down["slct_ct"] = celltype

        if direction in ["up", "both"] and len(up_deg) >= 5:
            enrich_res_up = fisher_enrich_test(up_deg, background, plist)
            if len(enrich_res_up) > 0:
                enrich_res_up["num_DEG"] = len(up_deg)
                enrich_res_up["direction_DEG"] = "upDEG"
                enrich_res_up["slct_ct"] = celltype

        # Combine results
        if enrich_res_up is not None and enrich_res_down is not None:
            enrich_list[celltype] = pd.concat(
                [enrich_res_up, enrich_res_down], ignore_index=True
            )
        elif enrich_res_up is not None:
            enrich_list[celltype] = enrich_res_up
        elif enrich_res_down is not None:
            enrich_list[celltype] = enrich_res_down

    if len(enrich_list) == 1:
        return list(enrich_list.values())[0]
    else:
        return enrich_list
