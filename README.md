# PertCurve

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

**PertCurve** is a trajectory-based perturbation scoring framework for single-cell perturbation data. It fits a principal curve from control cells to each perturbation, projects cells onto the fitted trajectory, and reports a normalized perturbation-response score for every cell.

This repository also includes **GEARS-PPC**, a score-aware GEARS workflow that uses PertCurve scores as continuous perturbation-strength signals during transcriptional response prediction.

## Overview

PertCurve summarizes heterogeneous single-cell perturbation responses with a pseudotime-like **Perturbation Principal Curve (PPC)** score:

- control-like cells receive low scores;
- cells farther along the fitted control-to-perturbation trajectory receive higher scores;
- each perturbation receives per-cell projection scores and perturbation-level distance summaries.

In the GEARS-PPC workflow, these scores are stored in `adata.obs["perturbation_score"]` and attached to each GEARS graph as `graph.pert_score`. The local GEARS fork can then condition its perturbation pathway on continuous response strength rather than using only a binary perturbation label.

## Key Features

- Fits perturbation-specific principal curves in PCA space.
- Produces per-cell normalized PPC scores, projection distances, and curve coordinates.
- Reports perturbation-level Wasserstein distance, KL divergence, curve length, endpoint distance, and cell counts.
- Provides plotting utilities for PC1/PC2 trajectory visualization.
- Includes downstream utilities for trajectory quality metrics, response trends, dose-response-style fitting, and response archetype summaries.
- Includes a local GEARS-PPC workflow for comparing baseline GEARS against PPC-aware perturbation prediction.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <this-repository-url>
cd PertCurve
pip install -e .
```

For development and tests:

```bash
pip install -e ".[test]"
pytest
```

For GEARS-PPC experiments, install PyTorch and PyTorch Geometric for your CUDA or CPU environment, then install the additional requirements and local GEARS fork:

```bash
pip install -r requirements-gears.txt
pip install -e src/GEARS-PPC
```

PyTorch Geometric wheels are environment-specific. If installation fails, follow the official PyG wheel selector for your Python, PyTorch, CUDA, and operating-system versions.

## Quick Start

Run PertCurve on an AnnData object with one perturbation label column and one control label:

```bash
python scripts/PertCurve/run_pertcurve.py \
  --input-h5ad path/to/processed_dataset.h5ad \
  --perturbation-col perturbation \
  --control-label control \
  --out-dir results/pertcurve \
  --n-pcs 50 \
  --n-bins 20 \
  --smoothing 0.3 \
  --plot
```

The `--plot` flag saves PNG trajectory plots. Without it, PertCurve writes only CSV outputs.

## Data

Associated data files are available on Zenodo:

- [https://zenodo.org/records/20643438](https://zenodo.org/records/20643438)

### Norman Example

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

### Replogle Example

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

## Output Files

For each run, `scripts/PertCurve/run_pertcurve.py` writes:

| Path | Description |
| --- | --- |
| `projections/projection_<perturbation>.csv` | Per-cell `normalized_pseudotime`, arc length, projection distance, and PC1/PC2 coordinates. |
| `perturbation_distance_stats.csv` | Perturbation-level Wasserstein distance, KL divergence, curve length, endpoint distance, and cell counts. |
| `figures/projection_<perturbation>.png` | Optional PC1/PC2 trajectory plot when `--plot` is enabled. |

These outputs can be used directly for trajectory analysis, downstream response modeling, or as the score source for GEARS-PPC.

## GEARS-PPC Workflow

GEARS-PPC expects a GEARS-formatted AnnData object with:

- `adata.obs["condition"]`: GEARS perturbation condition label, such as `TP73+ctrl`;
- `adata.obs["condition_name"]`: full perturbation name used by GEARS metadata;
- `adata.obs["perturbation_score"]`: the PertCurve/PPC score for each cell.

The Norman dataset in the companion data tree includes a scored example:

```text
../dataset/GEARS/norman/perturb_processed_with_scores.h5ad
```

If you need to rebuild the scored GEARS input from projection CSV files, use the dataset-side helper as a template:

```bash
cd /home/yhzhong/projects/singlecell/reverse-perturb/public_sources/dataset/GEARS/process_script
python assign_scores_to_h5ad.py
```

Review the path constants in that script before running it on a new dataset.

### Train

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

The script writes timestamped run directories under `results/gears_PPC/`, including:

- `manifest.json`: run configuration;
- `per_run/*.json`: metrics for each variant and seed;
- `models/<seed>_<variant>.pt`: saved checkpoints;
- `all_runs.csv`, `summary.json`, and paired baseline-versus-dual test results.

### Evaluate

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

### Plot

Plot baseline and PPC-aware predictions for selected perturbations:

```bash
python scripts/GEARS-PPC/plot_PPC_gears.py \
  --gears-root src/GEARS-PPC \
  --data-root ../dataset/GEARS \
  --dataset-name norman \
  --pretrained-model-dir results/gears_PPC/models/1_baseline.pt results/gears_PPC/models/1_dual.pt \
  --model-labels baseline dual \
  --perturbation TP73+ctrl \
  --device auto \
  --output-dir results/gears_PPC/figures
```

When two models are supplied, the plotting script saves an all-ground-truth-cell panel and, when scores are available, a high-score-cell panel.

## Python API

PertCurve modules can also be used directly from Python:

```python
from pertcurve.metrics import evaluate_trajectory_quality

metrics = evaluate_trajectory_quality(
    scores=df_scores["normalized_pseudotime"].to_numpy(),
    expression=adata_sub[:, de_genes].X,
    embedding=adata_sub.obsm["X_pca"],
)
```

Important modules include:

- `pertcurve.curve`: principal-curve fitting utilities;
- `pertcurve.projection`: projection and trajectory coordinate utilities;
- `pertcurve.scoring`: perturbation-level PPC scoring;
- `pertcurve.metrics`: trajectory quality and neighborhood-overlap metrics;
- `pertcurve.downstream`: response trend, Hill/log-logistic fitting, and response archetype utilities;
- `pertcurve.plotting`: PC1/PC2 projection figures.

## Repository Layout

```text
PertCurve/
|-- src/pertcurve/              # Core PertCurve implementation
|-- src/GEARS-PPC/              # Local score-aware GEARS fork
|-- src/mixscale-py/            # Bundled Mixscale reference code
|-- scripts/PertCurve/          # PertCurve scoring scripts
|-- scripts/GEARS-PPC/          # GEARS-PPC training, evaluation, and plotting scripts
|-- tests/                      # Unit tests
|-- results/                    # Example or generated outputs
|-- requirements.txt            # Core runtime requirements
|-- requirements-gears.txt      # GEARS-PPC requirements
`-- pyproject.toml              # Package metadata
```

Large raw datasets, long logs, and external reference repositories are not included. The command examples assume a companion dataset tree at `../dataset/`; override command-line paths when using a different layout.

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover core curve fitting, downstream summaries, multiple-testing utilities, trajectory metrics, and GEARS/PertCurve integration points.

## Notes

- `src/GEARS-PPC` is a local score-aware GEARS fork, not the upstream GEARS package.
- GEARS-PPC requires `adata.obs["perturbation_score"]` in score-aware paths.
- For leakage-free held-out evaluation, avoid using true test-cell PPC scores as test-time side information. The evaluation script assigns a constant score of `0.5` to non-baseline test graphs.
- The default command-line paths are examples. Check dataset paths before running workflows on a new machine or dataset.

## Citation

If you use PertCurve or GEARS-PPC in your work, please cite the corresponding project or manuscript when available.

## License

This project is distributed under the MIT license, as declared in `pyproject.toml`.
