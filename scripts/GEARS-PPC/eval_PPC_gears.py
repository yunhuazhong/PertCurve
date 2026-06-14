#!/usr/bin/env python3
"""Evaluate saved GEARS comparison checkpoints.

This script loads each model checkpoint from a comparison run directory,
rebuilds the dataset split, runs test-time evaluation, and aggregates
per-perturbation metrics across seeds.

Outputs:
- per_model_metrics.csv: one row per checkpoint with per-perturbation metrics
- averaged_per_perturbation_metrics.csv: mean/std across checkpoints
- averaged_per_perturbation_metrics.json: same aggregation in JSON form
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


CHECKPOINT_RE = re.compile(r"^(?P<seed>\d+)_(?P<variant>[A-Za-z0-9_\-]+)\.pt$")
NATURE_HEX_COLORS = [
    "#2664BF",
    "#34A89A",
    "#F69CA9",
    "#FBD399",
    "#AD95D1",
    "#FEA992",
]
PLOT_SUBGROUPS = ["combo_seen0", "combo_seen1", "combo_seen2", "unseen_single"]
PLOT_SUBGROUP_LABELS = {
    "combo_seen0": "0/2 seen",
    "combo_seen1": "1/2 seen",
    "combo_seen2": "2/2 seen",
    "unseen_single": "unseen single",
}


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    seed: int
    variant: str


def ensure_local_gears_import(gears_repo_root: Path) -> None:
    sys.path.insert(0, str(gears_repo_root))


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def flatten_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for pert, vals in metrics.items():
        for metric_name, value in vals.items():
            flat[f"{pert}.{metric_name}"] = float(value)
    return flat


def canonicalize_perturbation_name(perturbation: str) -> str:
    """Normalize perturbation labels so A+B and B+A collapse together.

    Keep singleton perturbations unchanged. For multi-part labels, use a
    deterministic canonical order that preserves "ctrl" at the end when
    present, while sorting the remaining perturbation names.
    """

    parts = [part.strip() for part in perturbation.split("+") if part.strip()]
    if len(parts) <= 1:
        return perturbation.strip()

    ctrl_parts = [part for part in parts if part.lower() == "ctrl"]
    other_parts = sorted(part for part in parts if part.lower() != "ctrl")
    return "+".join(other_parts + ctrl_parts)


def save_csv_rows(path: Path, rows: List[Dict], append: bool = False) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    write_header = True
    mode = "w"
    if append and path.exists():
        mode = "a"
        write_header = False
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_manifest(run_dir: Path) -> Dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def init_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper", rc={"figure.figsize": (6, 3)})
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 10
    return plt, sns


def load_plot_data(eval_dir: Path):
    import pandas as pd

    data_records = []
    per_run_dir = eval_dir / "per_run"
    if not per_run_dir.exists():
        per_run_dir = eval_dir / "eval_oracle" / "per_run"

    for variant in ["baseline", "dual"]:
        for path in sorted(per_run_dir.glob(f"{variant}_seed*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            seed = data.get("seed", 0)
            for subgroup, metrics in data.get("subgroup_metrics", {}).items():
                if subgroup not in PLOT_SUBGROUPS:
                    continue
                data_records.append(
                    {
                        "variant": "w/o PertCurve" if variant == "baseline" else "w/ PertCurve",
                        "seed": seed,
                        "subgroup": subgroup,
                        "MSE(DE)": metrics.get("mse_de"),
                        "Pearson(DE)": metrics.get("pearson_de"),
                        "MSE": metrics.get("mse"),
                        "Pearson": metrics.get("pearson"),
                        "MSE(DE)_high_score": metrics.get("mse_de_high_score"),
                        "Pearson(DE)_high_score": metrics.get("pearson_de_high_score"),
                        "MSE_high_score": metrics.get("mse_high_score"),
                        "Pearson_high_score": metrics.get("pearson_high_score"),
                    }
                )
    return pd.DataFrame(data_records)


def plot_metric_pair(
    df,
    output_dir: Path,
    filename_stem: str,
    left_metric: str,
    right_metric: str,
    left_title: str,
    right_title: str,
    left_ylabel: str,
    right_ylabel: str,
    left_ylim,
    right_ylim,
    left_yticks,
    right_yticks,
) -> None:
    plt, sns = init_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    fig.subplots_adjust(wspace=0.25)

    plot_kwargs = {
        "data": df,
        "x": "Subgroup",
        "hue": "variant",
        "palette": [NATURE_HEX_COLORS[0], NATURE_HEX_COLORS[1]],
        "errorbar": "se",
        "capsize": 0.12,
        "err_kws": {"linewidth": 1},
    }
    sns.barplot(y=left_metric, ax=axes[0], **plot_kwargs)
    axes[0].set_title(left_title)
    axes[0].set_ylabel(left_ylabel)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="", frameon=False, loc="upper center", fontsize=8)
    axes[0].set_ylim(*left_ylim)
    axes[0].set_yticks(left_yticks)
    axes[0].set_yticklabels(left_yticks)

    sns.barplot(y=right_metric, ax=axes[1], **plot_kwargs)
    axes[1].set_title(right_title)
    axes[1].set_ylabel(right_ylabel)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="", frameon=False, loc="upper center", fontsize=8)
    axes[1].set_ylim(*right_ylim)
    axes[1].set_yticks(right_yticks)
    axes[1].set_yticklabels(right_yticks)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{filename_stem}.svg"
    png_path = output_dir / f"{filename_stem}.png"
    plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=300, transparent=False)
    plt.close()
    print(f"Saved figures to {svg_path} and {png_path}")


def plot_comparison_outputs(eval_dir: Path) -> None:
    df = load_plot_data(eval_dir)
    if df.empty:
        print(f"No per-run subgroup metrics found for plotting under {eval_dir}")
        return

    df["Subgroup"] = df["subgroup"].map(PLOT_SUBGROUP_LABELS)
    output_dir = eval_dir / "performance"

    plot_metric_pair(
        df=df,
        output_dir=output_dir,
        filename_stem="subgroup_metrics_comparison",
        left_metric="MSE(DE)",
        right_metric="Pearson(DE)",
        left_title="MSE (DE)",
        right_title="Pearson Correlation (DE)",
        left_ylabel="MSE (DE)",
        right_ylabel="Pearson (DE)",
        left_ylim=(0.1, 0.7),
        right_ylim=(0.6, 1.0),
        left_yticks=[0.1, 0.3, 0.5, 0.7],
        right_yticks=[0.6, 0.7, 0.8, 0.9, 1],
    )
    plot_metric_pair(
        df=df,
        output_dir=output_dir,
        filename_stem="subgroup_metrics_comparison_all",
        left_metric="MSE",
        right_metric="Pearson",
        left_title="MSE",
        right_title="Pearson Correlation",
        left_ylabel="MSE",
        right_ylabel="Pearson",
        left_ylim=(0, 0.03),
        right_ylim=(0.94, 1.0),
        left_yticks=[0, 0.01, 0.02, 0.03],
        right_yticks=[0.94, 0.96, 0.98, 1.0],
    )
    plot_metric_pair(
        df=df,
        output_dir=output_dir,
        filename_stem="subgroup_metrics_comparison_high_score_de",
        left_metric="MSE(DE)_high_score",
        right_metric="Pearson(DE)_high_score",
        left_title="MSE (DE) (Strength > 0.5)",
        right_title="Pearson Correlation (DE) (Strength > 0.5)",
        left_ylabel="MSE (DE)",
        right_ylabel="Pearson (DE)",
        left_ylim=(0.2, 1.0),
        right_ylim=(0.6, 1.0),
        left_yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        right_yticks=[0.6, 0.7, 0.8, 0.9, 1.0],
    )
    plot_metric_pair(
        df=df,
        output_dir=output_dir,
        filename_stem="subgroup_metrics_comparison_high_score_all",
        left_metric="MSE_high_score",
        right_metric="Pearson_high_score",
        left_title="MSE (Strength > 0.5)",
        right_title="Pearson Correlation (Strength > 0.5)",
        left_ylabel="MSE",
        right_ylabel="Pearson",
        left_ylim=(0, 0.03),
        right_ylim=(0.92, 1.0),
        left_yticks=[0, 0.01, 0.02, 0.03],
        right_yticks=[0.92, 0.94, 0.96, 0.98, 1.0],
    )


def discover_checkpoints(models_dir: Path) -> List[CheckpointInfo]:
    checkpoints: List[CheckpointInfo] = []
    for path in sorted(models_dir.glob("*.pt")):
        m = CHECKPOINT_RE.match(path.name)
        if not m:
            continue
        checkpoints.append(
            CheckpointInfo(
                path=path,
                seed=int(m.group("seed")),
                variant=m.group("variant"),
            )
        )
    return checkpoints


def subgroup_summary(pert_metrics: Dict[str, Dict[str, float]], subgroup_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metric_keys = [
        "mse",
        "pearson",
        "mse_de",
        "pearson_de",
        "mse_high_score",
        "pearson_high_score",
        "mse_de_high_score",
        "pearson_de_high_score",
    ]
    for subgroup_name, perts in subgroup_dict.items():
        out[subgroup_name] = {}
        for metric in metric_keys:
            vals = [pert_metrics[p][metric] for p in perts if p in pert_metrics and metric in pert_metrics[p] and not np.isnan(pert_metrics[p][metric])]
            out[subgroup_name][metric] = float(np.mean(vals)) if vals else float("nan")
    return out


def canonicalize_per_pert_metrics(per_pert_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Merge duplicate perturbations that differ only by ordering.

    For example, RUNX1T1+ctrl and ctrl+RUNX1T1 are accumulated into the same
    canonical key RUNX1T1+ctrl before any downstream aggregation.
    """

    merged: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for pert, metrics in per_pert_metrics.items():
        canonical_pert = canonicalize_perturbation_name(pert)
        for metric_name, value in metrics.items():
            merged[canonical_pert][metric_name].append(float(value))

    output: Dict[str, Dict[str, float]] = {}
    for pert, metric_lists in merged.items():
        output[pert] = {}
        for metric_name, values in metric_lists.items():
            output[pert][metric_name] = float(np.mean(np.array(values, dtype=float)))
    return output


def build_dataset(data_root: Path, dataset_name: str, batch_size: int, seed: int, variant: str):
    from gears import PertData

    pert_data = PertData(str(data_root))
    pert_data.load(data_name=dataset_name)

    if variant == "baseline":
        remove_scores(pert_data)
    else:
        # attach_scores_per_cell(pert_data)
        for _, graphs in pert_data.dataset_processed.items():
            for graph in graphs:
                graph.pert_score = torch.tensor([[0.5]], dtype=torch.float32)
        # remove_scores(pert_data)

    pert_data.prepare_split(split="simulation", seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=128)
    return pert_data


def evaluate_checkpoint(
    checkpoint: CheckpointInfo,
    data_root: Path,
    dataset_name: str,
    batch_size: int,
    hidden_size: int,
    device: str,
    gears_repo_root: Path,
) -> Dict:
    ensure_local_gears_import(gears_repo_root)

    from gears import GEARS
    from gears.inference import compute_metrics, evaluate
    import gears.gears as gears_module
    import gears.model as dual_model_module
    # Match the training script's model selection convention.
        
    gears_module.GEARS_Model = dual_model_module.GEARS_Model
    run_device = device
    if run_device == "auto":
        run_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pert_data = build_dataset(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=batch_size,
        seed=checkpoint.seed,
        variant=checkpoint.variant,
    )
    model = GEARS(pert_data, device=run_device)
    model.model_initialize(hidden_size=hidden_size)

    state_dict = torch.load(checkpoint.path, map_location=model.device)
    model.best_model.load_state_dict(state_dict)
    model.best_model.to(model.device)
    model.best_model.eval()

    test_loader = pert_data.dataloader["test_loader"]
    test_res = evaluate(test_loader, model.best_model, model.config["uncertainty"], model.device)
    overall_metrics, per_pert_metrics = compute_metrics(test_res)
    per_pert_metrics = canonicalize_per_pert_metrics(per_pert_metrics)
    
    high_score_metrics = evaluate_high_score_subset_metrics(pert_data, test_res)
    high_score_metrics = canonicalize_per_pert_metrics(high_score_metrics)
    for pert, h_metrics in high_score_metrics.items():
        if pert in per_pert_metrics:
            per_pert_metrics[pert].update(h_metrics)
        else:
            per_pert_metrics[pert] = h_metrics
            
    subgroup_metrics = subgroup_summary(per_pert_metrics, pert_data.subgroup["test_subgroup"])
    strength_metrics = top50_perturbation_strength(model, pert_data, test_res)
    strength_metrics = canonicalize_per_pert_metrics(strength_metrics)
    
    for pert, s_metrics in strength_metrics.items():
        if pert in per_pert_metrics:
            per_pert_metrics[pert].update(s_metrics)
        else:
            per_pert_metrics[pert] = s_metrics
            
    # Augment overall_metrics
    overall = {k: float(v) for k, v in overall_metrics.items()}
    new_overall_keys = [
        "mse_high_score", "pearson_high_score", "mse_de_high_score", "pearson_de_high_score",
    ]
    for mk in new_overall_keys:
        vals = [pm[mk] for pm in per_pert_metrics.values() if mk in pm and not np.isnan(pm[mk])]
        overall[mk] = float(np.mean(vals)) if vals else float("nan")

    return {
        "checkpoint": checkpoint.path.name,
        "variant": checkpoint.variant,
        "seed": checkpoint.seed,
        "overall_metrics": overall,
        "per_pert_metrics": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in per_pert_metrics.items()},
        "subgroup_metrics": subgroup_metrics,
        "strength_metrics": strength_metrics,
    }


def remove_scores(pert_data) -> int:
    removed = 0
    for _, graphs in pert_data.dataset_processed.items():
        for graph in graphs:
            if hasattr(graph, "pert_score"):
                delattr(graph, "pert_score")
                removed += 1
    return removed


def attach_scores_per_cell(pert_data) -> int:
    if "perturbation_score" not in pert_data.adata.obs.columns:
        raise ValueError("adata.obs does not contain 'perturbation_score'")

    attached = 0
    obs = pert_data.adata.obs
    for cond, graphs in pert_data.dataset_processed.items():
        values = obs.loc[obs["condition"] == cond, "perturbation_score"].to_numpy(dtype=float)
        if len(values) == 0:
            values = np.zeros(1, dtype=float)
        repeat = max(1, len(graphs) // len(values))
        for idx, graph in enumerate(graphs):
            score_idx = min(len(values) - 1, idx // repeat)
            score_tensor = torch.tensor([[float(values[score_idx])]], dtype=torch.float32)
            graph.pert_score = score_tensor
            attached += 1
    return attached


def evaluate_high_score_subset_metrics(pert_data, test_res) -> Dict[str, Dict[str, float]]:
    adata = pert_data.adata
    pert2full = dict(adata.obs[["condition", "condition_name"]].values)
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    
    score_col = 'perturbation_score'
    has_scores = score_col in adata.obs.columns

    out: Dict[str, Dict[str, float]] = {}
    for pert in np.unique(test_res["pert_cat"]):
        if pert == "ctrl":
            continue
            
        p_idx = np.where(test_res["pert_cat"] == pert)[0]
        if len(p_idx) == 0:
            continue
            
        pred_mean = test_res["pred"][p_idx].mean(0)
        
        full_name = pert2full.get(pert, pert)
        
        true_mean = None
        if has_scores:
            query_mask = (adata.obs['condition'] == pert).values
            query_scores = adata.obs.loc[query_mask, score_col].values
            score_mask = query_scores > 0.5
            if np.sum(score_mask) > 0:
                truth_subset = adata[adata.obs['condition'] == pert].X.toarray()[score_mask]
                true_mean = truth_subset.mean(0)
                
        if true_mean is None:
            # If no scores or no cells > 0.5, skip or NaN
            out[pert] = {
                "mse_high_score": float("nan"),
                "pearson_high_score": float("nan"),
                "mse_de_high_score": float("nan"),
                "pearson_de_high_score": float("nan"),
            }
            continue
            
        # overall mse and pearson
        mse_high = float(np.mean((pred_mean - true_mean)**2))
        pearson_high = float(np.corrcoef(pred_mean, true_mean)[0, 1]) if np.std(pred_mean)>0 and np.std(true_mean)>0 else 0.0
        
        # de genes
        de_genes = adata.uns.get('top_non_dropout_de_20', {}).get(full_name, [])
        de_idx = [geneid2idx[g] for g in de_genes if g in geneid2idx]
        
        if len(de_idx) > 1:
            mse_de_high = float(np.mean((pred_mean[de_idx] - true_mean[de_idx])**2))
            pearson_de_high = float(np.corrcoef(pred_mean[de_idx], true_mean[de_idx])[0, 1])
        else:
            mse_de_high = float("nan")
            pearson_de_high = float("nan")

        if np.isnan(pearson_high):
            pearson_high = 0.0
        if np.isnan(pearson_de_high):
            pearson_de_high = 0.0

        out[pert] = {
            "mse_high_score": mse_high,
            "pearson_high_score": pearson_high,
            "mse_de_high_score": mse_de_high,
            "pearson_de_high_score": pearson_de_high,
        }
    return out


def top50_perturbation_strength(model, pert_data, test_res) -> Dict[str, Dict[str, float]]:
    """Compute top-50 perturbation-strength style metrics.

    This mirrors the GEARS plot_perturbation convention that focuses on the
    strongest perturbation-response genes. We use the top-50 DE genes from the
    per-perturbation annotations and summarize the absolute predicted-vs-truth
    deviation from control in that subset.
    """

    adata = pert_data.adata
    pert2full = dict(adata.obs[["condition", "condition_name"]].values)
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    ctrl_mask = (adata.obs["condition"] == "ctrl").values
    ctrl = np.asarray(np.mean(adata.X[ctrl_mask], axis=0)).ravel()

    out: Dict[str, Dict[str, float]] = {}
    for pert in np.unique(test_res["pert_cat"]):
        if pert == "ctrl":
            continue
        full_name = pert2full.get(pert, pert)
        rank_map = adata.uns.get("rank_genes_groups_cov_all", {})
        de_genes = rank_map.get(full_name, [])[:50]
        if de_genes is None or len(de_genes) == 0:
            continue
        de_idx = [geneid2idx[g] for g in de_genes if g in geneid2idx]
        if not de_idx:
            continue

        p_idx = np.where(test_res["pert_cat"] == pert)[0]
        pred_mean = test_res["pred"][p_idx].mean(0)[de_idx]
        true_mean = test_res["truth"][p_idx].mean(0)[de_idx]
        pred_delta = pred_mean - ctrl[de_idx]
        true_delta = true_mean - ctrl[de_idx]

        pearson_all = np.corrcoef(pred_mean, true_mean)[0, 1] if len(de_idx) > 1 else 0.0
        pearson_de = np.corrcoef(pred_delta, true_delta)[0, 1] if len(de_idx) > 1 else 0.0
        if np.isnan(pearson_all):
            pearson_all = 0.0
        if np.isnan(pearson_de):
            pearson_de = 0.0

        out[pert] = {
            "mse_top50_strength": float(np.mean((pred_mean - true_mean) ** 2)),
            "pearson_top50_strength": float(pearson_all),
            "mse_de_top50_strength": float(np.mean((pred_delta - true_delta) ** 2)),
            "pearson_de_top50_strength": float(pearson_de),
        }
    return out


def aggregate_per_perturbation(runs: List[Dict]) -> Dict:
    # Aggregate only the per-perturbation metrics requested by the user.
    # Canonicalize perturbation labels first so A+B and B+A are merged.
    by_variant: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        variant = run["variant"]
        for pert, metrics in run["per_pert_metrics"].items():
            canonical_pert = canonicalize_perturbation_name(pert)
            for metric_name in (
                "mse_de", "pearson_de", "mse", "pearson", 
                "mse_high_score", "pearson_high_score", "mse_de_high_score", "pearson_de_high_score",
            ):
                if metric_name in metrics:
                    by_variant[variant][canonical_pert][metric_name].append(float(metrics[metric_name]))

    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for variant, perts in by_variant.items():
        aggregated[variant] = {}
        for pert, metrics in perts.items():
            aggregated[variant][pert] = {}
            for metric_name, values in metrics.items():
                arr = np.array(values, dtype=float)
                aggregated[variant][pert][metric_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n": int(len(arr)),
                }
    return aggregated


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_results_dir = project_root / "results" / "gears_PPC"
    default_gears_root = project_root / "src" / "GEARS-PPC"
    default_data_root = project_root.parent / "dataset" / "GEARS"

    parser = argparse.ArgumentParser(description="Evaluate saved GEARS comparison checkpoints.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_results_dir,
        help="Comparison run directory containing models/ and manifest.json",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory (defaults to <run-dir>/models)",
    )
    parser.add_argument("--gears-root", type=Path, default=None, help="Override GEARS repo root")
    parser.add_argument("--data-root", type=Path, default=None, help="Override GEARS data root")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override dataset name")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override hidden size")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. cuda:0 or auto)")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append per-model results to CSV/JSONL outputs instead of overwriting",
    )

    parser.add_argument(
        "--output-dir", type=Path,
        default=default_results_dir / "eval",
        help="Directory to save aggregated evaluation outputs (CSV/JSON).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip subgroup comparison plots after evaluation.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)

    models_dir = (args.models_dir or (run_dir / "models")).resolve()
    if not models_dir.exists():
        raise FileNotFoundError(f"models directory not found: {models_dir}")

    gears_root_value = args.gears_root or manifest.get("gears_root") or default_gears_root
    if not gears_root_value:
        raise ValueError("Unable to determine GEARS root. Pass --gears-root or provide manifest.json.")
    gears_root = Path(gears_root_value).resolve()

    data_root_value = args.data_root or manifest.get("data_root") or default_data_root
    if not data_root_value:
        raise ValueError("Unable to determine data root. Pass --data-root or provide manifest.json.")
    data_root = Path(data_root_value).resolve()

    dataset_name = args.dataset_name or manifest.get("dataset_name", "norman")
    batch_size = args.batch_size or int(manifest.get("batch_size", 32))
    hidden_size = args.hidden_size or int(manifest.get("hidden_size", 64))
    device = args.device or manifest.get("device", "auto")

    checkpoints = discover_checkpoints(models_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files matching '<seed>_<variant>.pt' found in {models_dir}")

    all_runs: List[Dict] = []
    out_dir = args.output_dir or run_dir
    per_model_csv = out_dir / "eval_per_model_metrics.csv"
    per_model_jsonl = out_dir / "eval_per_model_metrics.jsonl"

    for ckpt in checkpoints:
        print(f"Evaluating {ckpt.path.name} ...", flush=True)
        result = evaluate_checkpoint(
            checkpoint=ckpt,
            data_root=data_root,
            dataset_name=dataset_name,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
            gears_repo_root=gears_root,
        )
        all_runs.append(result)
        
        # Save per-run specific summary mirroring baseline_seed1.json
        per_run_dir = out_dir / "per_run"
        per_run_dir.mkdir(parents=True, exist_ok=True)
        per_run_path = per_run_dir / f"{ckpt.variant}_seed{ckpt.seed}.json"
        
        # Omit 'per_pert_metrics' array in the quick summary
        summary_payload = {
            "variant": ckpt.variant,
            "seed": ckpt.seed,
            "overall_metrics": result["overall_metrics"],
            "subgroup_metrics": result["subgroup_metrics"],
        }
        per_run_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

        row = {
            "checkpoint": result["checkpoint"],
            "variant": result["variant"],
            "seed": result["seed"],
        }
        row.update({f"overall.{k}": v for k, v in result["overall_metrics"].items()})
        row.update(flatten_metrics(result["per_pert_metrics"]))
        save_csv_rows(per_model_csv, [row], append=True)

        with per_model_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        aggregated = aggregate_per_perturbation(all_runs)
        (out_dir / "averaged_per_perturbation_metrics.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "models_dir": str(models_dir),
                    "created_at": now_iso(),
                    "evaluated_checkpoints": len(all_runs),
                    "aggregated": aggregated,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        summary_rows = []
        for variant, perts in aggregated.items():
            for pert, metrics in perts.items():
                for metric_name, stat in metrics.items():
                    summary_rows.append(
                        {
                            "variant": variant,
                            "perturbation": pert,
                            "metric": metric_name,
                            "mean": stat["mean"],
                            "std": stat["std"],
                            "n": stat["n"],
                        }
                    )
        save_csv_rows(out_dir / "averaged_per_perturbation_metrics.csv", summary_rows)

    (out_dir / "eval_per_model_metrics.json").write_text(
        json.dumps(all_runs, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Saved evaluation outputs to {out_dir}")
    if not args.skip_plots:
        plot_comparison_outputs(out_dir)


if __name__ == "__main__":
    main()
