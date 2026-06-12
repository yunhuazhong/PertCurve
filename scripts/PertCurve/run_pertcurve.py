#!/usr/bin/env python
"""Run PertCurve scoring on an AnnData perturbation dataset."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pertcurve.io import ensure_dir, ensure_pca, list_perturbations, read_h5ad, safe_filename
from pertcurve.plotting import plot_projection
from pertcurve.scoring import score_perturbation


def main():
    args = parse_args()
    cfg = load_config(args.config)
    params = {**cfg, **{k: v for k, v in vars(args).items() if v is not None}}

    input_h5ad = params["input_h5ad"]
    perturbation_col = params["perturbation_col"]
    control_label = str(params["control_label"])
    out_dir = ensure_dir(params.get("out_dir", "pertcurve_results"))
    projections_dir = ensure_dir(out_dir / "projections")
    figures_dir = ensure_dir(out_dir / "figures")

    adata = read_h5ad(input_h5ad)
    X_pca_all = ensure_pca(
        adata,
        n_pcs=int(params.get("n_pcs", 50)),
        pca_key=params.get("pca_key", "X_pca"),
    )
    adata.obs[perturbation_col] = adata.obs[perturbation_col].astype(str)

    perturbations = list_perturbations(adata, perturbation_col, control_label)
    if params.get("perturbations"):
        requested = {str(p) for p in params["perturbations"]}
        perturbations = [p for p in perturbations if p in requested]

    stats_rows = []
    min_cells_per_group = int(params.get("min_cells_per_group", 5))

    for idx, pert in enumerate(perturbations, start=1):
        mask = adata.obs[perturbation_col].isin([control_label, pert]).to_numpy()
        labels = adata.obs.loc[mask, perturbation_col].to_numpy()
        n_ctrl = int((labels == control_label).sum())
        n_pert = int((labels != control_label).sum())
        if n_ctrl < min_cells_per_group or n_pert < min_cells_per_group:
            print(f"[{idx}/{len(perturbations)}] Skipping {pert}: too few cells.")
            continue

        X = X_pca_all[mask]
        cell_ids = adata.obs_names[mask].astype(str)
        print(f"[{idx}/{len(perturbations)}] Scoring {pert} ({n_ctrl} control, {n_pert} perturbed).")

        df_scores, stats, curve, _ = score_perturbation(
            X,
            labels,
            cell_ids=cell_ids,
            control_label=control_label,
            n_bins=int(params.get("n_bins", 20)),
            smoothing=float(params.get("smoothing", 0.5)),
            n_curve_points=int(params.get("n_curve_points", 100)),
        )

        filename = safe_filename(pert)
        df_scores.to_csv(projections_dir / f"projection_{filename}.csv", index=False)
        stats_rows.append({"perturbation": pert, **stats})

        if bool(params.get("plot", False)):
            plot_projection(
                X=X,
                labels=labels,
                curve=curve,
                scores=df_scores["normalized_pseudotime"].to_numpy(),
                control_label=control_label,
                title=f"{pert} (PertCurve)",
                out_path=figures_dir / f"projection_{filename}.png",
            )

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values("wasserstein_dist", ascending=False)
    stats_df.to_csv(out_dir / "perturbation_distance_stats.csv", index=False)
    print(f"Saved results to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PertCurve perturbation trajectory scoring.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--input-h5ad", dest="input_h5ad", type=str, default=None, help="Input AnnData .h5ad file.")
    parser.add_argument("--perturbation-col", dest="perturbation_col", type=str, default=None, help="Column in adata.obs with perturbation labels.")
    parser.add_argument("--control-label", dest="control_label", type=str, default=None, help="Control label in perturbation column.")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--pca-key", dest="pca_key", type=str, default=None, help="PCA key in adata.obsm.")
    parser.add_argument("--n-pcs", dest="n_pcs", type=int, default=None, help="Number of PCs to use.")
    parser.add_argument("--n-bins", dest="n_bins", type=int, default=None, help="Number of trajectory bins.")
    parser.add_argument("--smoothing", type=float, default=None, help="Spline smoothing value.")
    parser.add_argument("--n-curve-points", dest="n_curve_points", type=int, default=None, help="Number of points in fitted curve.")
    parser.add_argument("--min-cells-per-group", dest="min_cells_per_group", type=int, default=None, help="Minimum control and perturbed cells required.")
    parser.add_argument("--perturbations", nargs="+", default=None, help="Optional subset of perturbations to score.")
    parser.add_argument("--plot", action="store_true", default=None, help="Save PC1/PC2 trajectory plots.")
    return parser.parse_args()


def load_config(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
