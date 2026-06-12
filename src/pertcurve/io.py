"""I/O helpers for AnnData-based PertCurve workflows."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_h5ad(path):
    """Read an AnnData file with scanpy imported lazily."""
    import scanpy as sc

    return sc.read_h5ad(path)


def ensure_pca(adata, n_pcs=50, pca_key="X_pca"):
    """Return PCA coordinates, computing PCA when the requested key is missing."""
    import scanpy as sc

    if pca_key not in adata.obsm or adata.obsm[pca_key].shape[1] < n_pcs:
        sc.pp.pca(adata, n_comps=n_pcs)
        pca_key = "X_pca"
    return np.asarray(adata.obsm[pca_key])[:, :n_pcs]


def list_perturbations(adata, perturbation_col, control_label):
    """Return sorted non-control perturbation labels from AnnData.obs."""
    if perturbation_col not in adata.obs:
        raise ValueError(f"{perturbation_col!r} was not found in adata.obs.")
    values = adata.obs[perturbation_col].dropna().astype(str).unique().tolist()
    return sorted(value for value in values if value != str(control_label))


def safe_filename(value):
    """Convert arbitrary perturbation labels to filesystem-safe names."""
    value = str(value)
    value = re.sub(r"[^\w.+-]+", "_", value)
    return value.strip("_") or "perturbation"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
