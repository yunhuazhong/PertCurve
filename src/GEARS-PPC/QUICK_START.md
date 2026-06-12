# Quick Start Guide: GEARS with Perturbation Scores

## Overview
This guide helps you quickly integrate and use perturbation scores in the GEARS model.

## 1. Prerequisites

Ensure you have:
- ✅ H5ad file with `perturbation_score` column in `adata.obs`
- ✅ Modified GEARS package in `GEARS-with-score/`
- ✅ PyTorch and PyTorch Geometric installed

## 2. Quick Test

Verify the integration works:

```bash
cd /home/yhzhong/projects/singlecell/reverse-perturb/experiments/1210-curve-on-gears
python test_score_integration.py
```

Expected output: All 5 tests should pass ✓

## 3. Basic Training

Train with default settings (lambda=0.3):

```bash
python train_with_scores.py \
    --data_path ./perturb_processed_with_scores.h5ad \
    --epochs 20 \
    --device cuda
```

## 4. Lambda Comparison

Find optimal lambda value:

```bash
python train_with_scores.py \
    --data_path ./perturb_processed_with_scores.h5ad \
    --compare \
    --lambda_min 0.1 \
    --lambda_max 0.5 \
    --lambda_steps 5 \
    --epochs 20 \
    --device cuda
```

This tests lambda values: [0.1, 0.2, 0.3, 0.4, 0.5]

## 5. Custom Training

Train with specific lambda:

```bash
python train_with_scores.py \
    --data_path ./perturb_processed_with_scores.h5ad \
    --lambda 0.35 \
    --hidden_size 128 \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --device cuda
```

## 6. Using in Python

```python
import sys
sys.path.insert(0, './GEARS-with-score')

from gears import PertData, GEARS

# Load data with scores
pert_data = PertData('./data')
pert_data.load(data_path='./perturb_processed_with_scores.h5ad')

# Prepare split
pert_data.prepare_split(split='simulation', seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# Initialize model
gears_model = GEARS(pert_data, device='cuda')
gears_model.model_initialize(hidden_size=64)

# Set score weight (optional, default is 0.3)
gears_model.model.score_lambda = 0.35

# Train
gears_model.train(epochs=20, lr=1e-3)

# Evaluate
test_metrics = gears_model.evaluate(pert_data.dataloader['test_loader'])
print(f"Test MSE: {test_metrics['mse']:.4f}")
print(f"Test Pearson: {test_metrics['pearson']:.4f}")
```

## 7. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `score_lambda` | 0.3 | Weight for score contribution (0.0 = disabled) |
| `hidden_size` | 64 | Dimension of hidden layers |
| `batch_size` | 32 | Training batch size |
| `epochs` | 20 | Number of training epochs |
| `lr` | 1e-3 | Learning rate |

## 8. Understanding score_lambda

- **0.0**: Scores not used (baseline GEARS)
- **0.1-0.2**: Subtle score influence
- **0.3**: Balanced (recommended starting point)
- **0.4-0.5**: Strong score influence
- **>0.5**: May overfit to scores

## 9. Monitoring Training

Check logs for:
```
Set model.score_lambda = 0.3
Epoch 1/20: train_loss=X.XX, val_loss=X.XX
...
Test MSE: X.XXXX
Test Pearson: X.XXXX
```

## 10. Results Location

```
results/
├── lambda_0.3/
│   ├── model.pt              # Trained model weights
│   └── metrics.json          # Performance metrics
└── lambda_comparison_YYYYMMDD_HHMMSS/
    ├── lambda_0.1/
    ├── lambda_0.2/
    ├── ...
    └── comparison_summary.json  # All results
```

## 11. Troubleshooting

**Issue**: "perturbation_score column not found"
```bash
# Re-run score assignment
python assign_scores_to_h5ad.py
```

**Issue**: "CUDA out of memory"
```bash
# Reduce batch size
python train_with_scores.py --batch_size 16
```

**Issue**: Model not using scores
```python
# Verify scores are present
import scanpy as sc
adata = sc.read_h5ad('perturb_processed_with_scores.h5ad')
print('perturbation_score' in adata.obs.columns)  # Should be True
```

## 12. Validation

Compare with baseline (no scores):

```bash
# Train without scores (lambda=0)
python train_with_scores.py --lambda 0.0 --output_dir ./results/baseline

# Train with scores (lambda=0.3)
python train_with_scores.py --lambda 0.3 --output_dir ./results/with_scores

# Compare metrics.json in both directories
```

## 13. Next Steps

1. ✓ Run tests
2. ✓ Train with default lambda
3. ✓ Compare with baseline (lambda=0)
4. ✓ Optimize lambda value
5. ✓ Analyze predictions
6. ✓ Tune other hyperparameters

## 14. Support Files

- `test_score_integration.py` - Verify integration
- `train_with_scores.py` - Training script
- `GEARS_SCORE_OPTIMIZATION.md` - Detailed documentation
- `assign_scores_to_h5ad.py` - Generate score file

## 15. Citation

If using this modified GEARS model, cite:
1. Original GEARS paper
2. Your perturbation score methodology
3. This optimization work

---

For detailed information, see `GEARS_SCORE_OPTIMIZATION.md`
