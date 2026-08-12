#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash 03_run_fig4f_single_model_pca.sh \
#   EXPERIMENT_ROOT \
#   PROJECT_ROOT \
#   OUT_DIR
#
# Default representative checkpoint:
# Task 4 / LOSO / seed 42 / fold 0
#
# Override with:
# SEED=43 FOLD=10 ...

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 EXPERIMENT_ROOT PROJECT_ROOT OUT_DIR"
  exit 2
fi

EXPERIMENT_ROOT="$1"
PROJECT_ROOT="$2"
OUT_DIR="$3"

SEED=${SEED:-42}
FOLD=${FOLD:-0}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
STANDARDIZE=${STANDARDIZE:-0}

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

STD_FLAG="--no-standardize"
if [[ "$STANDARDIZE" == "1" ]]; then
  STD_FLAG="--standardize"
fi

echo "===== Fig.4f SINGLE-MODEL feature visualization ====="
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Project root:    $PROJECT_ROOT"
echo "Output:          $OUT_DIR"
echo "Checkpoint:      Task4 / LOSO / seed_$SEED / fold_$FOLD"
echo "Batch size:      $BATCH_SIZE"
echo "Num workers:     $NUM_WORKERS"
echo "Standardize PCA: $STANDARDIZE"
echo

python "$DIR/01_export_single_model_subject_features.py" \
  --experiment_root "$EXPERIMENT_ROOT" \
  --project_root "$PROJECT_ROOT" \
  --out_dir "$OUT_DIR/features" \
  --seed "$SEED" \
  --fold "$FOLD" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \

FEATURE_CSV="$OUT_DIR/features/task4_single_model_seed${SEED}_fold${FOLD}_subject_features.csv"

python "$DIR/02_plot_pca_single_model.py" \
  --feature_csv "$FEATURE_CSV" \
  --out_dir "$OUT_DIR/pca" \
  $STD_FLAG

echo
echo "[DONE]"
echo "Feature CSV:"
echo "  $FEATURE_CSV"
echo "PCA:"
echo "  $OUT_DIR/pca/Fig4f_PCA_single_model.png"
