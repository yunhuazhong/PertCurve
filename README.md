# PertCurve and GEARS-PPC

This directory contains the public code for PertCurve, a Perturbation Principal Curve (PPC) method for scoring single-cell perturbation responses, and GEARS-PPC, a score-aware GEARS workflow that uses PertCurve/PPC scores during perturbation-response prediction.

PertCurve fits a low-dimensional principal curve between control cells and each perturbation, projects cells onto that curve, and reports a normalized pseudotime-like perturbation score. In the GEARS experiments, this score is used as a per-cell perturbation strength signal, stored as `adata.obs["perturbation_score"]`, and passed to a modified GEARS model through each graph's `pert_score` field.

## What Is PPC?

PPC is the PertCurve perturbation-principal-curve score. It summarizes where each cell lies along a fitted control-to-perturbation trajectory:

- cells close to the control centroid receive low scores;
- cells farther along the perturbation trajectory receive higher scores;
- each perturbation receives both per-cell scores and distribution-level distance summaries.

This gives GEARS more than a binary perturbation label. Instead of treating every cell assigned to the same perturbation as equally strong, GEARS-PPC can condition its perturbation pathway on a continuous response score.

## Repository Layout

```text
public_sources/code/
|-- src/pertcurve/              # Core PertCurve/PPC implementation
|-- src/GEARS-PPC/              # Local GEARS fork with PPC score gating
|-- src/mixscale-py/            # Bundled Mixscale reference code
|-- scripts/PertCurve/          # CLI for generating PPC scores
|-- scripts/GEARS-PPC/          # Train, evaluate, and plot GEARS-PPC runs
|-- tests/                      # Unit tests for PertCurve components
`-- results/                    # Example/generated outputs
```

Large datasets are not part of this code package. The companion dataset tree is expected at `../dataset/` when using the default command-line paths.

## Installation

Install the PertCurve package from this directory:

```bash
cd /home/yhzhong/projects/singlecell/reverse-perturb/public_sources/code
pip install -e .
```

For tests:

```bash
pip install -e ".[test]"
pytest
```

For GEARS-PPC, install PyTorch and PyTorch Geometric for your CUDA or CPU environment, then install the local GEARS fork if needed:

```bash
pip install -r requirements-gears.txt
pip install -e src/GEARS-PPC
```

PyTorch Geometric installation is environment-specific; follow the wheel selector from the PyG documentation if the generic requirements install does not match your CUDA version.

## Generate PPC Scores

Run PertCurve on an AnnData object with one perturbation label column and one control label. The script writes one score table per perturbation plus summary metrics.

Norman-style example:

```bash
python scripts/PertCurve/run_pertcurve.py \
  --input-h5ad ../dataset/processed/norman_filtered_processed.h5ad \
  --perturbation-col perturbation \
  --control-label control \
  --out-dir results/norman_pertcurve \
  --n-pcs 50 \
  --n-bins 20 \
  --smoothing 0.3 \
  --plot
```

Replogle-style example:

```bash
python scripts/PertCurve/run_pertcurve.py \
  --input-h5ad ../dataset/processed/replogle_filtered_processed.h5ad \
  --perturbation-col gene \
  --control-label non-targeting \
  --out-dir results/replogle_pertcurve \
  --n-pcs 80 \
  --n-bins 20 \
  --smoothing 0.5 \
  --plot
```

The `--plot` flag is required to save PNG trajectory figures. Without it, only CSV outputs are produced.

## PertCurve Outputs

For each run, `scripts/PertCurve/run_pertcurve.py` writes:

- `projections/projection_<perturbation>.csv`: per-cell `normalized_pseudotime`, arc length, projection distance, and PC1/PC2 coordinates.
- `perturbation_distance_stats.csv`: perturbation-level Wasserstein distance, KL divergence, curve length, endpoint distance, and cell counts.
- `figures/projection_<perturbation>.png`: optional PC1/PC2 trajectory plots when `--plot` is enabled.

These outputs can be used directly for trajectory analysis, downstream response modeling, or as the score source for GEARS-PPC.

## Apply PPC Scores To GEARS

GEARS-PPC expects a GEARS-formatted AnnData file with:

- `adata.obs["condition"]`: GEARS perturbation condition label, such as `TP73+ctrl`;
- `adata.obs["condition_name"]`: full perturbation name used by GEARS metadata;
- `adata.obs["perturbation_score"]`: the PPC score for each cell.

The Norman dataset in the companion data tree already includes a scored file:

```text
../dataset/GEARS/norman/perturb_processed_with_scores.h5ad
```

If you need to rebuild it from projection CSVs, use the dataset-side helper as a template:

```bash
cd /home/yhzhong/projects/singlecell/reverse-perturb/public_sources/dataset/GEARS/process_script
python assign_scores_to_h5ad.py
```

That script maps PertCurve `normalized_pseudotime` values into `adata.obs["perturbation_score"]` and writes a GEARS-ready `.h5ad`. Check its path constants before running it on a new dataset.

## How GEARS-PPC Uses The Score

The local GEARS fork in `src/GEARS-PPC` modifies the GEARS perturbation pathway:

- the training script attaches `adata.obs["perturbation_score"]` to each graph as `graph.pert_score`;
- the PPC model reads `data.pert_score` in `src/GEARS-PPC/gears/model.py`;
- a residual score gate scales the perturbation embedding, allowing the model to attenuate or amplify the perturbation effect based on PPC strength;
- the `baseline` variant removes scores, while the `dual` variant keeps score-aware gating.

This makes the comparison explicit: vanilla GEARS versus GEARS with continuous PertCurve/PPC response information.

## Train GEARS-PPC

The main training script compares `baseline` and `dual` variants by default:

```bash
python scripts/GEARS-PPC/train_PPC_gears.py \
  --gears-root src/GEARS-PPC \
  --data-root ../dataset/GEARS \
  --dataset-name norman \
  --variants baseline,dual \
  --seeds 1 \
  --epochs 20 \
  --batch-size 32 \
  --hidden-size 64 \
  --device auto \
  --output-dir results/gears_PPC
```

Outputs are written to a timestamped run directory under `results/gears_PPC/`:

- `manifest.json`: run configuration;
- `per_run/*.json`: metrics for each variant and seed;
- `models/<seed>_<variant>.pt`: saved checkpoints;
- `all_runs.csv`, `summary.json`, and paired baseline-vs-dual test results.

## Evaluate Checkpoints

Evaluate saved checkpoints and aggregate per-perturbation metrics:

```bash
python scripts/GEARS-PPC/eval_PPC_gears.py \
  --run-dir results/gears_PPC/<run_dir> \
  --gears-root src/GEARS-PPC \
  --data-root ../dataset/GEARS \
  --dataset-name norman \
  --device auto \
  --output-dir results/gears_PPC/eval
```

Evaluation writes per-model metrics, averaged per-perturbation summaries, high-score subset metrics, and JSON/CSV outputs suitable for downstream plotting.

## Plot GEARS-PPC Predictions

Plot baseline and PPC-aware predictions for selected perturbations:

```bash
python scripts/GEARS-PPC/plot_PPC_gears.py \
  --gears-root src/GEARS-PPC \
  --data-root ../dataset/GEARS \
  --dataset-name norman \
  --pretrained-model-dir results/gears_PPC/models/1_baseline.pt results/gears_PPC/models/1_dual.pt \
  --model-labels baseline dual \
  --perturbations TP73+ctrl \
  --device auto \
  --output-dir results/gears_PPC/figures
```

When two models are supplied, the plotting script saves an all-ground-truth-cell panel and, if scores are available, a high-score-cell panel.

## Analysis Utilities

The core `pertcurve` package also includes modules for quality checks and downstream response analysis:

- `pertcurve.metrics`: smoothness, reconstruction, mutual information, and kNN-overlap metrics for trajectory quality.
- `pertcurve.downstream`: binned response trends, Hill/log-logistic response fitting, and response archetype summaries.
- `pertcurve.plotting`: PC1/PC2 projection figures for PertCurve runs.

Example:

```python
from pertcurve.metrics import evaluate_trajectory_quality

metrics = evaluate_trajectory_quality(
    scores=df_scores["normalized_pseudotime"].to_numpy(),
    expression=adata_sub[:, de_genes].X,
    embedding=adata_sub.obsm["X_pca"],
)
```

## Notes

- This release intentionally excludes large raw datasets, long logs, and external reference repositories.
- `src/GEARS-PPC` is a local score-aware GEARS fork, not the upstream GEARS package.
- GEARS-PPC requires `adata.obs["perturbation_score"]`; missing scores will raise an error in the score-aware path.
- For leakage-free held-out evaluation, avoid using true test-cell PPC scores as test-time side information. The evaluation script assigns a constant score of `0.5` to non-baseline test graphs.
