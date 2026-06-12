#!/usr/bin/env python3
"""Plot a GEARS perturbation result from trained or pretrained models.

This utility allows plotting predictions from one or more models (e.g. baseline and dual) 
alongside a single ground-truth boxplot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_local_gears_import(gears_repo_root: Path) -> None:
    sys.path.insert(0, str(gears_repo_root))


def build_pert_data(data_root: Path, dataset_name: str, seed: int):
    from gears import PertData
    pert_data = PertData(str(data_root))
    pert_data.load(data_name=dataset_name)
    pert_data.prepare_split(split="simulation", seed=seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data


def load_model(pert_data, device: str, pretrained_model_path: Path):
    from gears import GEARS

    model = GEARS(pert_data, device=device)

    # Support either a checkpoint directory (legacy) or a direct .pt file.
    checkpoint_dir = pretrained_model_path if pretrained_model_path.is_dir() else pretrained_model_path.parent
    config_path = checkpoint_dir / "config.pkl"
    state_path = pretrained_model_path if pretrained_model_path.suffix == ".pt" else checkpoint_dir / "model.pt"

    if config_path.exists() and state_path.exists() and pretrained_model_path.is_dir():
        model.load_pretrained(str(checkpoint_dir))
        return model

    # Fallback for analysis-only runs where only the torch checkpoint exists.
    model.model_initialize()
    if state_path.exists():
        import torch

        state_dict = torch.load(state_path, map_location=torch.device("cpu"))
        if next(iter(state_dict))[:7] == "module.":
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict
        model.model.load_state_dict(state_dict)
        model.best_model = model.model
        print(f"Loaded checkpoint weights from {state_path}")
        return model

    raise FileNotFoundError(f"No usable checkpoint found at {pretrained_model_path}.")


def plot_multiple_models(models: list, labels: list, perturbation: str, output_path: Path, genes_to_plot=None, sort_genes: bool = True):
    import seaborn as sns
    import matplotlib.lines as mlines
    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    legend_name_map = {
        "baseline": "w/o PertCurve",
        "dual": "w/ PertCurve",
    }
    display_labels = [legend_name_map.get(label, label) for label in labels]

    model0 = models[0]
    adata = model0.adata
    gene2idx = model0.node_map
    cond2name = dict(adata.obs[['condition', 'condition_name']].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    if genes_to_plot is not None:
        gene_name_to_ensg = {gene_raw2id[i]: i for i in adata.var.index.values}
        ordered_genes = [g for g in genes_to_plot if g in gene_name_to_ensg]
        if sort_genes:
            ordered_genes = sorted(ordered_genes)
        genes_ENSG = [gene_name_to_ensg[g] for g in ordered_genes]
        de_idx = [gene2idx[gene_raw2id[i]] for i in genes_ENSG]
        genes = np.array(ordered_genes)
        de_idx = np.array(de_idx)
    else:
        de_genes = [gene_raw2id[i] for i in adata.uns['top_non_dropout_de_20'][cond2name[perturbation]]]
        if sort_genes:
            de_genes = sorted(de_genes)
        de_idx = [gene2idx[g] for g in de_genes]
        genes = np.array(de_genes)
    
    truth = adata[adata.obs.condition == perturbation].X.toarray()[:, de_idx]
    
    query_ = [q for q in perturbation.split('+') if q != 'ctrl']
    
    ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values
    truth = truth - ctrl_means
    
    preds = []
    for m in models:
        pred_m = m.predict([query_])['_'.join(query_)][de_idx]
        preds.append(pred_m - ctrl_means)
        
    score_col = 'perturbation_score'
    has_scores = score_col in adata.obs.columns
    if has_scores:
        query_mask = (adata.obs['condition'] == perturbation).values
        query_scores = adata.obs.loc[query_mask, score_col].values
        score_mask = query_scores > 0.5
        truth_subset = truth[score_mask]
    else:
        truth_subset = None
        
    colors = ['red', 'blue', 'orange', 'purple', 'cyan']

    def _plot_and_save_single_panel(panel_truth, panel_title, out_path_single):
        fig, ax = plt.subplots(figsize=[9, 4.8], dpi=600)
        ax.set_title(panel_title)
        ax.boxplot(panel_truth, showfliers=False, showmeans=True,
                   meanprops=dict(marker='o', markerfacecolor='white',
                                  markeredgecolor='black', markersize=5),
                   medianprops=dict(linewidth=0))
        
        for m_idx, (m_pred, label) in enumerate(zip(preds, labels)):
            c = colors[m_idx % len(colors)]
            for i in range(m_pred.shape[0]):
                ax.scatter(i + 1, m_pred[i], color=c, s=50, zorder=3)
                
        ax.axhline(0, linestyle="dashed", color='green')
        ax.set_xticks(range(1, len(genes) + 1))
        ax.set_xticklabels(genes, rotation=90)
        ax.set_ylabel("", labelpad=10)
        ax.tick_params(axis='x', which='major', pad=5)
        ax.tick_params(axis='y', which='major', pad=5)
        
        # Explicitly build legend handles to avoid grey styles from default seaborn markers
        handles = [
            mlines.Line2D([], [], markerfacecolor=colors[i % len(colors)],
                          markeredgecolor=colors[i % len(colors)], color='w',
                          marker='o', linestyle='None', markersize=8, label=display_labels[i])
            for i in range(len(models))
        ]
        # right-top
        ax.legend(handles=handles, facecolor='white', framealpha=1, edgecolor='black', frameon=True, loc='upper right', prop={'size': 12})
        
        sns.despine(ax=ax)
        fig.tight_layout()
        out_path_single.parent.mkdir(parents=True, exist_ok=True)
        fig.patch.set_facecolor('white')
        fig.savefig(out_path_single, bbox_inches='tight', dpi=300, facecolor='white', transparent=False)
        plt.close(fig)

    # perturbation may be 'X+ctrl', but we want the title to just be 'X' for simplicity
    perturbation_title = perturbation.replace('+ctrl', '')

    out_path_all = output_path.with_name(f"{output_path.stem}_all_GT{output_path.suffix}")
    _plot_and_save_single_panel(truth, f"{perturbation_title} (all GT cells)", out_path_all)
    
    if truth_subset is not None:
        out_path_sub = output_path.with_name(f"{output_path.stem}_high_score{output_path.suffix}")
        _plot_and_save_single_panel(truth_subset, f"{perturbation_title} (Perturbation Strength > 0.5)", out_path_sub)


def save_plot(models, labels, perturbation: str, output_path: Path) -> None:
    choice = None

    if len(models) == 1:
        # Compatibility with original single-model execution
        result = models[0].plot_perturbation(perturbation, genes_to_plot=choice, sort_genes=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if result is not None:
            result.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig = plt.gcf()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return result
    else:
        plot_multiple_models(models, labels, perturbation, output_path, genes_to_plot=choice, sort_genes=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot GEARS perturbation predictions.")

    project_root = Path(__file__).resolve().parents[2]
    default_results_dir = project_root / "results" / "gears_PPC"
    default_model_paths = [
        default_results_dir / "models" / "1_baseline.pt",
        default_results_dir / "models" / "1_dual.pt",
    ]
    parser.add_argument("--gears-root", type=Path, default=project_root / "src" / "GEARS-PPC")
    parser.add_argument("--data-root", type=Path, default=project_root.parent / "dataset" / "GEARS")
    parser.add_argument("--dataset-name", type=str, default="norman")
    
    parser.add_argument(
        "--pretrained-model-dir", type=Path, nargs="+", dest="model_dirs",
        default=default_model_paths,
        help="Directory, .pt checkpoint, or list of checkpoints containing trained GEARS models.",
    )
    parser.add_argument(
        "--model-labels", type=str, nargs="+", default=["baseline", "dual"],
        help="Labels for the models if multiple are provided",
    )
    parser.add_argument(
        "--perturbation", type=str, default="TP73+ctrl",
    )
    parser.add_argument(
        "--perturbations", nargs="+",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=default_results_dir / "figures",
    )
    args = parser.parse_args()

    if not args.model_dirs:
        parser.error("You must specify at least one model directory with --pretrained-model-dir")

    perturbations = args.perturbations if args.perturbations else [args.perturbation]

    ensure_local_gears_import(args.gears_root)
    pert_data = build_pert_data(args.data_root, args.dataset_name, args.seed)
    
    models = [load_model(pert_data, args.device, md) for md in args.model_dirs]
    
    labels = args.model_labels
    if len(labels) < len(models):
        for i in range(len(labels), len(models)):
            labels.append(f"Model {i+1}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for perturbation in perturbations:
        per_output = args.output_dir / f"{perturbation}.png"
        save_plot(models, labels, perturbation, per_output)
        if len(models) > 1:
            print(f"Saved separate perturbation plots to {per_output.parent} as _all_GT.png and _high_score.png")
        else:
            print(f"Saved perturbation plot to {per_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
