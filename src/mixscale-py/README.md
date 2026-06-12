# Mixscale (Python)

Mixscale is a Python package designed to analyze CRISPR interference (CRISPRi) based Perturb-seq data. It can quantify the heterogeneity of perturbation strength in each cell and improve the statistical power when doing differential expression (DE) analysis. It also provides functions for downstream analyses including decomposition, permutation test, gene set enrichment test, etc.

This is a Python port of the original R package [Mixscale](https://github.com/satijalab/Mixscale).

## Dependencies

This package depends on several Python packages:
- numpy
- scipy
- pandas
- scanpy
- anndata
- scikit-learn
- matplotlib
- seaborn
- statsmodels

## Installation

You can install the package using pip:

```bash
pip install mixscale
```

Or install from source:

```bash
git clone https://github.com/satijalab/Mixscale
cd mixscale-py
pip install -e .
```

## Quick Start

```python
import mixscale as mx
import scanpy as sc

# Load your AnnData object
adata = sc.read_h5ad("your_perturbseq_data.h5ad")

# Calculate perturbation scores
mx.tl.run_mixscale(
    adata,
    labels="gene",
    nt_class_name="NT",
    layer="counts",
    split_by="celltype"
)

# Run weighted differential expression test
de_results = mx.tl.run_weighted_de(
    adata,
    labels="gene",
    nt_class_name="NT",
    split_by="celltype"
)

# Perform PCA-based permutation test
perm_results = mx.tl.pca_permtest(
    zscore_matrix,
    k=5,
    num_iter=200
)

# Visualize perturbation scores
mx.pl.ridge_plot(
    adata,
    labels="gene",
    nt_class_name="NT",
    split_by="celltype",
    prtb=["GENE1", "GENE2"]
)
```

## Main Functions

### Perturbation Scoring
- `mx.tl.run_mixscale()`: Calculate perturbation scores for each cell

### Differential Expression
- `mx.tl.run_weighted_de()`: Weighted DE test using perturbation scores
- `mx.tl.get_fold_change()`: Calculate log-fold-change

### Decomposition
- `mx.tl.pca_permtest()`: PCA-based permutation test
- `mx.tl.get_sig_genes()`: Extract significant genes from permutation test

### Enrichment Analysis
- `mx.tl.de_enrich()`: Wrapper for DE and enrichment test
- `mx.tl.fisher_enrich_test()`: Fisher's exact test for enrichment
- `mx.tl.rbo()`: Rank biased overlap for gene set enrichment

### Visualization
- `mx.pl.ridge_plot()`: Ridge plot for perturbation score distribution
- `mx.pl.expression_score_plot()`: Compare expression vs perturbation scores

## Documentation

Full documentation is available at: https://satijalab.github.io/Mixscale/

## Other Resources

* Our preprint is available at https://www.biorxiv.org/content/10.1101/2024.01.29.576933v2.
* Processed data from our paper: https://doi.org/10.5281/zenodo.14518762
* Raw fastq files: GEO accession [GSE281048](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281048)

## Citation

If you use Mixscale in your research, please cite:
```
[Citation to be added]
```

## License

MIT License
