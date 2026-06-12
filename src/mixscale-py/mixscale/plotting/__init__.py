"""Visualization module for Mixscale."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Optional, List, Union
import warnings


def ridge_plot(
    adata: AnnData,
    labels: str = "gene",
    nt_class_name: str = "NT",
    split_by: Optional[str] = None,
    prtb: Optional[List[str]] = None,
    slct_split_by: Optional[List[str]] = None,
    facet_wrap: Optional[str] = None,
    figsize: tuple = (10, 6),
    **kwargs,
) -> plt.Figure:
    """
    Ridge plot for perturbation score distribution.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with mixscale scores
    labels : str
        Column in adata.obs with target gene labels (default: 'gene')
    nt_class_name : str
        Non-targeting class name (default: 'NT')
    split_by : str, optional
        Column for cell type/condition
    prtb : list, optional
        Perturbations to plot
    slct_split_by : list, optional
        Selected conditions to plot
    facet_wrap : str, optional
        Facet by 'gene' or 'split.by'
    figsize : tuple
        Figure size (default: (10, 6))

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if scores exist
    if "mixscale_scores" not in adata.uns:
        raise ValueError("Mixscale scores not found. Run run_mixscale() first.")

    if "mixscale_score" not in adata.obs.columns:
        raise ValueError("mixscale_score column not found in adata.obs")

    # Get score data
    prtb_score = adata.uns["mixscale_scores"]

    # Handle split_by
    if split_by is None:
        splits = ["con1"]
        adata.obs["_split"] = "con1"
        split_by = "_split"
    else:
        splits = sorted(adata.obs[split_by].unique())

    if slct_split_by is not None:
        splits = [s for s in splits if s in slct_split_by]

    # Collect scores
    all_scores = []

    for celltype in splits:
        for prtb_gene in prtb:
            if prtb_gene not in prtb_score or celltype not in prtb_score[prtb_gene]:
                continue

            scores = prtb_score[prtb_gene][celltype]

            if isinstance(scores, pd.DataFrame) and "pvec" in scores.columns:
                tmp = scores[["pvec", "gene"]].copy()
                tmp["celltype"] = celltype
                tmp["PRTB_group"] = prtb_gene
                all_scores.append(tmp)

    if not all_scores:
        print(f"No score data found for {prtb} perturbations.")
        return None

    all_scores_df = pd.concat(all_scores, ignore_index=True)
    all_scores_df["status"] = nt_class_name
    all_scores_df.loc[all_scores_df["gene"] != nt_class_name, "status"] = "perturbed"

    # Create plot
    if facet_wrap == "split.by":
        n_facets = len(all_scores_df["PRTB_group"].unique())
        fig, axes = plt.subplots(1, n_facets, figsize=(figsize[0] * n_facets, figsize[1]))

        if n_facets == 1:
            axes = [axes]

        for idx, prtb_gene in enumerate(sorted(all_scores_df["PRTB_group"].unique())):
            subset = all_scores_df[all_scores_df["PRTB_group"] == prtb_gene]

            for celltype in sorted(subset["celltype"].unique()):
                for status in ["perturbed", nt_class_name]:
                    data = subset[
                        (subset["celltype"] == celltype) & (subset["status"] == status)
                    ]["pvec"]

                    if len(data) > 0:
                        axes[idx].hist(
                            data,
                            bins=30,
                            alpha=0.5,
                            label=f"{celltype}_{status}",
                            density=True,
                        )

            axes[idx].set_xlabel("Perturbation Score")
            axes[idx].set_ylabel("Density")
            axes[idx].set_title(prtb_gene)
            axes[idx].legend()

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for prtb_gene in sorted(all_scores_df["PRTB_group"].unique()):
            for status in ["perturbed", nt_class_name]:
                data = all_scores_df[
                    (all_scores_df["PRTB_group"] == prtb_gene)
                    & (all_scores_df["status"] == status)
                ]["pvec"]

                if len(data) > 0:
                    ax.hist(
                        data,
                        bins=30,
                        alpha=0.5,
                        label=f"{prtb_gene}_{status}",
                        density=True,
                    )

        ax.set_xlabel("Perturbation Score")
        ax.set_ylabel("Density")
        ax.set_title("Perturbation Score Distribution")
        ax.legend()

    plt.tight_layout()
    # save to 'ridge_plot_{prtb.join("_")}.png'
    plt.savefig(f"ridge_plot_{'_'.join(prtb)}.png", dpi=300)
    return fig


def expression_score_plot(
    adata: AnnData,
    gene_name: str,
    labels: str = "gene",
    nt_class_name: str = "NT",
    split_by: Optional[str] = None,
    nbin: int = 10,
    layer: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Compare expression level vs perturbation scores.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    gene_name : str
        Gene to plot expression for
    labels : str
        Column with target gene labels (default: 'gene')
    nt_class_name : str
        Non-targeting class name (default: 'NT')
    split_by : str, optional
        Column for cell type/condition
    nbin : int
        Number of bins for scores (default: 10)
    layer : str, optional
        Expression layer to use
    figsize : tuple
        Figure size (default: (8, 6))

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if "mixscale_score" not in adata.obs.columns:
        raise ValueError("mixscale_score not found. Run run_mixscale() first.")

    # Get expression data
    if layer is None:
        if hasattr(adata, "X"):
            gene_idx = list(adata.var_names).index(gene_name)
            expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, "toarray") else adata.X[:, gene_idx]
        else:
            raise ValueError("No expression data found")
    else:
        gene_idx = list(adata.var_names).index(gene_name)
        expr = adata.layers[layer][:, gene_idx]

    # Filter to perturbation target
    mask = adata.obs[labels] == gene_name
    scores = adata.obs.loc[mask, "mixscale_score"].values
    expr_sub = expr[mask]

    # Bin scores
    bins = pd.qcut(scores, q=nbin, labels=False, duplicates="drop")

    # Calculate mean expression per bin
    bin_means = []
    for b in range(nbin):
        if b in bins:
            bin_means.append(np.mean(expr_sub[bins == b]))
        else:
            bin_means.append(np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(nbin), bin_means, marker="o", linestyle="-")
    ax.set_xlabel("Perturbation Score Percentile Bin")
    ax.set_ylabel(f"Mean {gene_name} Expression")
    ax.set_title(f"Expression vs Perturbation Score: {gene_name}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# def volcano_plot(
#     de_results: pd.DataFrame,
#     logfc_col: str = "logfoldchanges",
#     pval_col: str = "pvals_adj",
#     gene_col: str = "names",
#     logfc_threshold: float = 0.5,
#     pval_threshold: float = 0.05,
#     figsize: tuple = (8, 6),
#     highlight_genes: Optional[List[str]] = None,
# ) -> plt.Figure:
#     """
#     Volcano plot for DE results.

#     Parameters
#     ----------
#     de_results : pd.DataFrame
#         DE test results
#     logfc_col : str
#         Column name for log fold change (default: 'logfoldchanges')
#     pval_col : str
#         Column name for p-values (default: 'pvals_adj')
#     gene_col : str
#         Column name for gene names (default: 'names')
#     logfc_threshold : float
#         Log fold change threshold (default: 0.5)
#     pval_threshold : float
#         P-value threshold (default: 0.05)
#     figsize : tuple
#         Figure size (default: (8, 6))
#     highlight_genes : list, optional
#         Genes to highlight

#     Returns
#     -------
#     plt.Figure
#         Matplotlib figure object
#     """
#     fig, ax = plt.subplots(figsize=figsize)

#     # Calculate -log10(pval)
#     log_pval = -np.log10(de_results[pval_col])

#     # Color points
#     colors = np.where(
#         (np.abs(de_results[logfc_col]) > logfc_threshold)
#         & (de_results[pval_col] < pval_threshold),
#         "red",
#         "gray",
#     )

#     ax.scatter(
#         de_results[logfc_col], log_pval, c=colors, alpha=0.5, s=10, edgecolors="none"
#     )

#     # Add threshold lines
#     ax.axhline(-np.log10(pval_threshold), color="blue", linestyle="--", alpha=0.5)
#     ax.axvline(logfc_threshold, color="blue", linestyle="--", alpha=0.5)
#     ax.axvline(-logfc_threshold, color="blue", linestyle="--", alpha=0.5)

#     # Highlight specific genes
#     if highlight_genes is not None:
#         for gene in highlight_genes:
#             if gene in de_results[gene_col].values:
#                 idx = de_results[gene_col] == gene
#                 ax.scatter(
#                     de_results.loc[idx, logfc_col],
#                     -np.log10(de_results.loc[idx, pval_col]),
#                     c="orange",
#                     s=50,
#                     edgecolors="black",
#                     linewidths=1,
#                     zorder=10,
#                 )
#                 ax.text(
#                     de_results.loc[idx, logfc_col].values[0],
#                     -np.log10(de_results.loc[idx, pval_col].values[0]),
#                     gene,
#                     fontsize=8,
#                 )

#     ax.set_xlabel("Log2 Fold Change")
#     ax.set_ylabel("-Log10 Adjusted P-value")
#     ax.set_title("Volcano Plot")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     return fig
