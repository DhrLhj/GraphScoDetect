#!/usr/bin/env bash
set -euo pipefail
# Usage:
# bash 03_run_sensor_loso_exact.sh \
#   /data/duanshitong/scoliosis/new_label_v5_selected_protocols \
#   /data/duanshitong/scoliosis/new_label_v5_selected_protocol_experiment \
#   /data/duanshitong/scoliosis/best_stgcn_all_protocols_20260726 \
#   /data/duanshitong/scoliosis/new_label_v5_sensor_loso_exact

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 BASE_OUT_ROOT BASE_CODE_DIR PROJECT_ROOT SENSOR_OUT_ROOT"
  exit 2
fi
BASE_OUT_ROOT=$(realpath "$1")
BASE_CODE_DIR=$(realpath "$2")
PROJECT_ROOT=$(realpath "$3")
SENSOR_OUT_ROOT="$4"
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SEEDS_STR=${SEEDS:-"42 43 44"}
NUM_WORKERS=${NUM_WORKERS:-0}
AMP=${AMP:-1}
UNIAXIAL_DEFINITION=${UNIAXIAL_DEFINITION:-image}
OVERWRITE=${OVERWRITE:-0}
LIMIT_FOLDS=${LIMIT_FOLDS:-0}

read -r -a SEED_ARGS <<< "$SEEDS_STR"
AMP_FLAG="--no-amp"; [[ "$AMP" == "1" ]] && AMP_FLAG="--amp"
OW=""; [[ "$OVERWRITE" == "1" ]] && OW="--overwrite"

BASE_TRAIN="$BASE_CODE_DIR/03_train_all_protocols.py"
BASE_SUMMARY="$BASE_CODE_DIR/04_summarize_results.py"

echo "===== EXACT-v5 LOSO SENSOR EXPERIMENT ====="
echo "Baseline output: $BASE_OUT_ROOT"
echo "Canonical trainer: $BASE_TRAIN"
echo "Canonical summary: $BASE_SUMMARY"
echo "Sensor output: $SENSOR_OUT_ROOT"
echo "Seeds: ${SEED_ARGS[*]}"
echo "NUM_WORKERS=$NUM_WORKERS AMP=$AMP"
echo "NOTE: epochs/batch/lr/weight_decay/dropout/balanced_loss are read from completed baseline LOSO run_config."
echo "NOTE: All channels is reused from baseline and is not retrained."
echo

python "$DIR/00_check_baseline_loso_config.py" \
  --base_out_root "$BASE_OUT_ROOT" \
  --tasks 2 4 \
  --seeds "${SEED_ARGS[@]}" \
  --save_json "$SENSOR_OUT_ROOT/baseline_loso_config.json"

python "$DIR/01_train_sensor_experiments_loso.py" \
  --base_out_root "$BASE_OUT_ROOT" \
  --base_train_script "$BASE_TRAIN" \
  --project_root "$PROJECT_ROOT" \
  --out_root "$SENSOR_OUT_ROOT" \
  --tasks 2 4 \
  --seeds "${SEED_ARGS[@]}" \
  --num_workers "$NUM_WORKERS" \
  --uniaxial_definition "$UNIAXIAL_DEFINITION" \
  --limit_folds "$LIMIT_FOLDS" \
  $AMP_FLAG $OW

python "$DIR/02_summarize_sensor_experiments_loso.py" \
  --base_out_root "$BASE_OUT_ROOT" \
  --sensor_out_root "$SENSOR_OUT_ROOT" \
  --base_summary_script "$BASE_SUMMARY" \
  --seeds "${SEED_ARGS[@]}"

echo
echo "[DONE] $SENSOR_OUT_ROOT/summary/sensor_loso_exact_summary_auroc.xlsx"
echo "[SANITY] $SENSOR_OUT_ROOT/summary/00_baseline_match_check.csv"
