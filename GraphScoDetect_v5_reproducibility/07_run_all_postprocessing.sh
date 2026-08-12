#!/usr/bin/env bash
set -euo pipefail

# Run the main post-hoc analyses after training.
# Usage:
#   bash 07_run_all_postprocessing.sh OUT_ROOT

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 OUT_ROOT"
  exit 2
fi

OUT_ROOT=$(realpath "$1")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN_OUT="$OUT_ROOT/main_protocols"
SENSOR_OUT="$OUT_ROOT/sensor_loso"

python "$REPO_ROOT/postprocessing/01_calculate_task4_macro_auprc.py" \
  --summary_dir "$MAIN_OUT/summary" \
  --out_dir "$OUT_ROOT/analysis/task4_auprc"

python "$REPO_ROOT/postprocessing/02_analyze_task4_misclassifications.py" \
  --summary_dir "$MAIN_OUT/summary" \
  --out_dir "$OUT_ROOT/analysis/task4_misclassification"

if [[ -d "$SENSOR_OUT/results" ]]; then
  python "$REPO_ROOT/postprocessing/03_analyze_sensor_by_curve_location.py" \
    --base_out_root "$MAIN_OUT" \
    --sensor_out_root "$SENSOR_OUT" \
    --out_dir "$OUT_ROOT/analysis/sensor_by_curve_location" \
    --seeds 42 43 44
fi

python "$REPO_ROOT/postprocessing/04_analyze_curve_group_statistics.py" \
  --subject_csv "$MAIN_OUT/summary/07_subject_ensemble_predictions.csv" \
  --protocol loso \
  --out_dir "$OUT_ROOT/analysis/curve_group_statistics"

if [[ -f "$MAIN_OUT/results/task4/loso/seed_42/fold_0/model.pt" ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  SEED=42 FOLD=0 BATCH_SIZE=256 NUM_WORKERS=4 STANDARDIZE=0 \
  bash "$REPO_ROOT/visualization/03_run_fig4f_single_model_pca.sh" \
    "$MAIN_OUT" \
    "$REPO_ROOT" \
    "$OUT_ROOT/analysis/fig4f_single_model"
else
  echo "[WARN] PCA skipped: Task-4 LOSO seed42/fold0 model.pt not found."
fi

echo "[DONE] Postprocessing outputs: $OUT_ROOT/analysis"
