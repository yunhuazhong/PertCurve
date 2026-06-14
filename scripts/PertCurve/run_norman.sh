
ROOT="/home/yhzhong/projects/singlecell/reverse-perturb/public_sources"
CODE_DIR="$ROOT/code"
INPUT_H5AD="$ROOT/dataset/processed/norman_filtered_processed.h5ad"
OUT_DIR="$CODE_DIR/results/norman_pertcurve"

cd "$CODE_DIR"
mkdir -p "$OUT_DIR"

python scripts/PertCurve/run_pertcurve.py \
    --input-h5ad "$INPUT_H5AD" \
    --perturbation-col perturbation \
    --control-label control \
    --out-dir "$OUT_DIR" \
    --pca-key X_pca \
    --n-pcs 50 \
    --n-bins 20 \
    --smoothing 0.3 \
    --n-curve-points 100 \
    --min-cells-per-group 5 \
    --plot