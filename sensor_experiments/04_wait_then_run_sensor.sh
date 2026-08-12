#!/usr/bin/env bash
set -euo pipefail

# Wait for an already-running baseline training PID to finish, then launch
# exact-v5 LOSO sensor ablation in the background.
#
# Usage:
# bash 04_wait_then_run_sensor.sh \
#   BASE_PID \
#   BASE_OUT_ROOT \
#   BASE_CODE_DIR \
#   PROJECT_ROOT \
#   SENSOR_OUT_ROOT
#
# Example:
# bash 04_wait_then_run_sensor.sh \
#   123456 \
#   /data/duanshitong/scoliosis/new_label_v5_selected_protocols_weights_auroc \
#   /data/duanshitong/scoliosis/new_label_v5_selected_protocol_experiment \
#   /data/duanshitong/scoliosis/best_stgcn_all_protocols_20260726 \
#   /data/duanshitong/scoliosis/new_label_v5_sensor_loso_auroc

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 BASE_PID BASE_OUT_ROOT BASE_CODE_DIR PROJECT_ROOT SENSOR_OUT_ROOT"
  exit 2
fi

BASE_PID="$1"
BASE_OUT_ROOT=$(realpath "$2")
BASE_CODE_DIR=$(realpath "$3")
PROJECT_ROOT=$(realpath "$4")
SENSOR_OUT_ROOT="$5"

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

CHECK_INTERVAL=${CHECK_INTERVAL:-300}
SEEDS_STR=${SEEDS:-"42 43 44"}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-1}
UNIAXIAL_DEFINITION=${UNIAXIAL_DEFINITION:-image}

mkdir -p "$SENSOR_OUT_ROOT/nohup_logs"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$SENSOR_OUT_ROOT/nohup_logs/wait_then_sensor_${STAMP}.log"

{
  echo "[$(date)] Monitoring baseline PID=$BASE_PID"
  echo "Check interval: ${CHECK_INTERVAL}s"

  while kill -0 "$BASE_PID" 2>/dev/null; do
    echo "[$(date)] Baseline training is still running..."
    sleep "$CHECK_INTERVAL"
  done

  echo "[$(date)] Baseline PID has exited."
  echo "[$(date)] Checking completed LOSO baseline configuration..."

  # This will fail early if LOSO outputs do not exist.
  python "$DIR/00_check_baseline_loso_config.py" \
    --base_out_root "$BASE_OUT_ROOT" \
    --tasks 2 4 \
    --seeds $SEEDS_STR \
    --save_json "$SENSOR_OUT_ROOT/baseline_loso_config.json"

  echo "[$(date)] Starting LOSO sensor ablation + AUROC."

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  SEEDS="$SEEDS_STR" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  UNIAXIAL_DEFINITION="$UNIAXIAL_DEFINITION" \
  bash "$DIR/03_run_sensor_experiments_loso.sh" \
    "$BASE_OUT_ROOT" \
    "$BASE_CODE_DIR" \
    "$PROJECT_ROOT" \
    "$SENSOR_OUT_ROOT"

  echo "[$(date)] Sensor experiment finished."
} > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$SENSOR_OUT_ROOT/nohup_logs/wait_then_sensor_${STAMP}.pid"
echo "Monitor/launcher PID: $PID"
echo "Log: $LOG"
echo "Watch with: tail -f $LOG"
