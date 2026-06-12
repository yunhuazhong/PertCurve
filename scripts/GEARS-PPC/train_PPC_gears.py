#!/usr/bin/env python3
"""
Run controlled GEARS comparison experiments in this workspace.

Supported variants:
- baseline: remove perturbation score input
- dual: residual-gated score scaling on perturbation pathway
- score_only: optional diagnostic variant using model_mod1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import torch
from scipy.stats import ttest_rel


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_local_gears_import(gears_repo_root: Path) -> None:
    # Force imports to use the project-local GEARS code.
    if not (gears_repo_root / "gears").is_dir():
        raise FileNotFoundError(
            f"Could not find a local GEARS package under {gears_repo_root}. "
            "Pass --gears-root pointing to the directory that contains gears/."
        )
    sys.path.insert(0, str(gears_repo_root))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        # GEARS default preprocessing uses num_samples=1, but keep this generic.
        repeat = max(1, len(graphs) // len(values))
        for idx, graph in enumerate(graphs):
            score_idx = min(len(values) - 1, idx // repeat)
            score_tensor = torch.tensor([[float(values[score_idx])]], dtype=torch.float32)
            graph.pert_score = score_tensor
            attached += 1
    return attached


def subgroup_summary(pert_metrics: Dict[str, Dict[str, float]], subgroup_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metric_keys = ["mse", "pearson", "mse_de", "pearson_de"]
    for subgroup_name, perts in subgroup_dict.items():
        out[subgroup_name] = {}
        for metric in metric_keys:
            vals = [pert_metrics[p][metric] for p in perts if p in pert_metrics and metric in pert_metrics[p]]
            out[subgroup_name][metric] = float(np.mean(vals)) if vals else float("nan")
    return out


def flatten_subgroup_metrics(subgroups: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for subgroup_name, metrics in subgroups.items():
        for metric_name, value in metrics.items():
            flat[f"{subgroup_name}.{metric_name}"] = float(value)
    return flat


def run_single(
    variant: str,
    seed: int,
    data_root: Path,
    dataset_name: str,
    batch_size: int,
    hidden_size: int,
    epochs: int,
    lr: float,
    device: str,
    model_save_dir: Path = None,
) -> Dict:
    set_seed(seed)

    from gears import GEARS, PertData
    from gears.inference import compute_metrics, evaluate
    import gears.gears as gears_module
    import gears.model as dual_model_module

    gears_module.GEARS_Model = dual_model_module.GEARS_Model

    start = time.time()
    pert_data = PertData(str(data_root))
    pert_data.load(data_name=dataset_name)

    if variant == "baseline":
        score_count = remove_scores(pert_data)
        score_mode = "removed"
    else:
        score_count = attach_scores_per_cell(pert_data)
        score_mode = "per_cell"

    pert_data.prepare_split(split="simulation", seed=seed)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=128)

    run_device = device
    if run_device == "auto":
        run_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = GEARS(pert_data, device=run_device)
    model.model_initialize(hidden_size=hidden_size)
    model.train(epochs=epochs, lr=lr)

    if model_save_dir is not None:
        model_path = model_save_dir / f"{seed}_{variant}.pt"
        torch.save(model.best_model.state_dict(), model_path)

    test_loader = pert_data.dataloader["test_loader"]
    test_res = evaluate(test_loader, model.best_model, model.config["uncertainty"], model.device)
    overall_metrics, per_pert_metrics = compute_metrics(test_res)
    subgroup_metrics = subgroup_summary(per_pert_metrics, pert_data.subgroup["test_subgroup"])

    elapsed = time.time() - start
    return {
        "variant": variant,
        "seed": seed,
        "device": run_device,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "lr": lr,
        "score_mode": score_mode,
        "score_count": score_count,
        "overall_metrics": {k: float(v) for k, v in overall_metrics.items()},
        "subgroup_metrics": subgroup_metrics,
        "elapsed_sec": float(elapsed),
        "timestamp": now_iso(),
    }


def aggregate_runs(runs: List[Dict]) -> Dict:
    by_variant: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        by_variant[run["variant"]].append(run)

    summary: Dict[str, Dict] = {}
    for variant, v_runs in by_variant.items():
        flat_rows = []
        for run in v_runs:
            row = {}
            row.update({f"overall.{k}": v for k, v in run["overall_metrics"].items()})
            row.update(flatten_subgroup_metrics(run["subgroup_metrics"]))
            flat_rows.append(row)

        metric_names = sorted(flat_rows[0].keys()) if flat_rows else []
        stats = {}
        for metric in metric_names:
            arr = np.array([r[metric] for r in flat_rows], dtype=float)
            stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
            }
        summary[variant] = stats
    return summary


def paired_tests(runs: List[Dict], a: str, b: str) -> Dict:
    by_key = {(r["variant"], r["seed"]): r for r in runs}
    common_seeds = sorted(
        set(r["seed"] for r in runs if r["variant"] == a)
        .intersection(set(r["seed"] for r in runs if r["variant"] == b))
    )
    if not common_seeds:
        return {"error": f"No shared seeds between {a} and {b}"}

    keys = set()
    for seed in common_seeds:
        ra = by_key[(a, seed)]
        rb = by_key[(b, seed)]
        flat_a = {}
        flat_a.update({f"overall.{k}": v for k, v in ra["overall_metrics"].items()})
        flat_a.update(flatten_subgroup_metrics(ra["subgroup_metrics"]))
        flat_b = {}
        flat_b.update({f"overall.{k}": v for k, v in rb["overall_metrics"].items()})
        flat_b.update(flatten_subgroup_metrics(rb["subgroup_metrics"]))
        keys.update(set(flat_a.keys()).intersection(set(flat_b.keys())))

    out = {"variant_a": a, "variant_b": b, "seeds": common_seeds, "metrics": {}}
    for key in sorted(keys):
        arr_a = []
        arr_b = []
        for seed in common_seeds:
            ra = by_key[(a, seed)]
            rb = by_key[(b, seed)]
            flat_a = {f"overall.{k}": v for k, v in ra["overall_metrics"].items()}
            flat_a.update(flatten_subgroup_metrics(ra["subgroup_metrics"]))
            flat_b = {f"overall.{k}": v for k, v in rb["overall_metrics"].items()}
            flat_b.update(flatten_subgroup_metrics(rb["subgroup_metrics"]))
            if key in flat_a and key in flat_b:
                arr_a.append(flat_a[key])
                arr_b.append(flat_b[key])

        arr_a_np = np.array(arr_a, dtype=float)
        arr_b_np = np.array(arr_b, dtype=float)
        if len(arr_a_np) >= 2 and len(arr_a_np) == len(arr_b_np):
            stat = ttest_rel(arr_a_np, arr_b_np)
            p_value = float(stat.pvalue)
        else:
            p_value = float("nan")

        out["metrics"][key] = {
            "a_mean": float(np.mean(arr_a_np)),
            "b_mean": float(np.mean(arr_b_np)),
            "delta_b_minus_a": float(np.mean(arr_b_np - arr_a_np)),
            "p_value_paired_ttest": p_value,
            "n": int(len(arr_a_np)),
        }
    return out


def save_csv_rows(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Run GEARS comparison experiments.")
    parser.add_argument("--gears-root", type=Path, default=project_root / "src" / "GEARS-PPC")
    parser.add_argument("--data-root", type=Path, default=project_root.parent / "dataset" / "GEARS")
    parser.add_argument("--dataset-name", type=str, default="norman")
    # Default protocol follows the current project decision:
    # compare vanilla GEARS (baseline) vs score-augmented GEARS (dual).
    parser.add_argument("--variants", type=str, default="baseline,dual")
    parser.add_argument("--seeds", type=str, default="1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/yhzhong/projects/singlecell/reverse-perturb/public_sources/code/results/gears_PPC"))
    args = parser.parse_args()

    ensure_local_gears_import(args.gears_root)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = run_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": now_iso(),
        "gears_root": str(args.gears_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "dataset_name": args.dataset_name,
        "variants": variants,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "device": args.device,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    all_runs: List[Dict] = []
    for variant in variants:
        for seed in seeds:
            print(f"\n=== Running variant={variant} seed={seed} ===", flush=True)
            result = run_single(
                variant=variant,
                seed=seed,
                data_root=args.data_root,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                lr=args.lr,
                device=args.device,
                model_save_dir=models_dir,
            )
            all_runs.append(result)
            out_file = per_run_dir / f"{variant}_seed{seed}.json"
            out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            om = result["overall_metrics"]
            print(
                "overall:",
                f"mse={om['mse']:.6f}",
                f"pearson={om['pearson']:.6f}",
                f"mse_de={om['mse_de']:.6f}",
                f"pearson_de={om['pearson_de']:.6f}",
                f"elapsed={result['elapsed_sec']:.1f}s",
                flush=True,
            )

    summary = aggregate_runs(all_runs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    flat_rows = []
    for run in all_runs:
        row = {
            "variant": run["variant"],
            "seed": run["seed"],
            "elapsed_sec": run["elapsed_sec"],
            "overall.mse": run["overall_metrics"]["mse"],
            "overall.pearson": run["overall_metrics"]["pearson"],
            "overall.mse_de": run["overall_metrics"]["mse_de"],
            "overall.pearson_de": run["overall_metrics"]["pearson_de"],
        }
        row.update(flatten_subgroup_metrics(run["subgroup_metrics"]))
        flat_rows.append(row)
    save_csv_rows(run_dir / "all_runs.csv", flat_rows)

    compare_pairs = [("baseline", "dual")]
    for a, b in compare_pairs:
        if a in variants and b in variants:
            compare = paired_tests(all_runs, a, b)
            out_name = f"paired_test_{a}_vs_{b}.json"
            (run_dir / out_name).write_text(
                json.dumps(compare, indent=2) + "\n", encoding="utf-8"
            )

    print("\nSaved outputs to:", run_dir, flush=True)


if __name__ == "__main__":
    main()
