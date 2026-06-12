# Mixscale Python Tutorial

This notebook demonstrates how to use the Mixscale Python package for analyzing Perturb-seq data.

## Installation

```python
# Install from source
!pip install -e /path/to/mixscale-py
```

## Load Libraries

```python
import mixscale as mx
import scanpy as sc
import pandas as pd
import numpy as np

# Set plotting parameters
sc.settings.set_figure_params(dpi=80, facecolor='white')
```

## Load Data

```python
# Load your Perturb-seq data
adata = sc.read_h5ad("your_perturbseq_data.h5ad")

# The data should have:
# - adata.obs['gene']: perturbation target labels
# - adata.obs['celltype']: cell type labels (optional)
# - adata.layers['counts']: raw count data

print(f"Data shape: {adata.shape}")
print(f"Perturbations: {adata.obs['gene'].unique()}")
```

## Calculate Perturbation Scores

```python
# Run Mixscale to calculate perturbation scores
mx.tl.run_mixscale(
    adata,
    labels='gene',
    nt_class_name='NT',  # Name of non-targeting control
    layer='counts',
    split_by='celltype',  # Optional: account for cell type differences
    max_de_genes=100,
    min_de_genes=5,
    logfc_threshold=0.25,
    verbose=True
)

# Scores are now stored in adata.obs['mixscale_score']
print(f"Score range: {adata.obs['mixscale_score'].min():.2f} to {adata.obs['mixscale_score'].max():.2f}")
```

## Visualize Perturbation Scores

```python
# Ridge plot showing score distribution
fig = mx.pl.ridge_plot(
    adata,
    labels='gene',
    nt_class_name='NT',
    split_by='celltype',
    prtb=['GENE1', 'GENE2', 'GENE3'],  # Genes to visualize
    figsize=(12, 6)
)
```

## PCA-Based Permutation Test

For downstream analysis, you can use PCA-based permutation testing to identify gene signatures.

```python
# First, prepare a Z-score matrix (genes x conditions)
# This should contain Z-scores from DE tests

# Example: Create a dummy Z-score matrix
zscore_matrix = pd.DataFrame(
    np.random.randn(1000, 20),  # 1000 genes x 20 conditions
    index=[f'gene_{i}' for i in range(1000)],
    columns=[f'condition_{i}' for i in range(20)]
)

# Run PCA permutation test
perm_results = mx.tl.pca_permtest(
    zscore_matrix,
    k=5,  # Number of top PCs
    num_iter=200,
    row_filtering_pval=0.05,
    verbose=True
)

# Extract significant genes
sig_genes = mx.tl.get_sig_genes(
    perm_results,
    k=5,
    perm_pval_thres=0.05,
    ori_pval_thres=1.67e-6,  # Bonferroni-corrected threshold
    collapse=True
)

print(f"Upregulated genes: {len(sig_genes['upDEGs'])}")
print(f"Downregulated genes: {len(sig_genes['downDEGs'])}")
```

## Gene Set Enrichment Analysis

```python
# Define gene sets
gene_sets = {
    'pathway1': ['gene_1', 'gene_5', 'gene_10', 'gene_15'],
    'pathway2': ['gene_2', 'gene_7', 'gene_12', 'gene_17'],
    'pathway3': ['gene_3', 'gene_8', 'gene_13', 'gene_18'],
}

# Run enrichment test
from mixscale.tools.enrichment import fisher_enrich_test

enrichment_results = fisher_enrich_test(
    input_list=sig_genes['upDEGs'][:100],
    background=zscore_matrix.index.tolist(),
    go_term_db=gene_sets
)

print(enrichment_results.head())
```

## Expression vs Score Analysis

```python
# Compare expression levels across perturbation score bins
fig = mx.pl.expression_score_plot(
    adata,
    gene_name='target_gene',
    labels='gene',
    nbin=10,
    figsize=(8, 6)
)
```

## Save Results

```python
# Save updated AnnData object
adata.write_h5ad("perturbseq_with_mixscale_scores.h5ad")

# Export scores to CSV
scores_df = pd.DataFrame({
    'cell_id': adata.obs_names,
    'perturbation': adata.obs['gene'],
    'mixscale_score': adata.obs['mixscale_score']
})
scores_df.to_csv("mixscale_scores.csv", index=False)
```

## Advanced: Custom DE Analysis

For more advanced users, you can access the detailed score information:

```python
# Access detailed score information
detailed_scores = adata.uns['mixscale_scores']

# Example: Get scores for a specific perturbation and cell type
prtb = 'GENE1'
celltype = 'celltype1'

if prtb in detailed_scores and celltype in detailed_scores[prtb]:
    scores = detailed_scores[prtb][celltype]
    print(f"Scores for {prtb} in {celltype}:")
    print(scores.head())
```

## Summary

This tutorial covered:
1. Loading Perturb-seq data
2. Calculating Mixscale perturbation scores
3. Visualizing score distributions
4. PCA-based permutation testing
5. Gene set enrichment analysis
6. Saving and exporting results

For more information, see the [documentation](https://satijalab.github.io/Mixscale/).
