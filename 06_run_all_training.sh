#!/usr/bin/env bash
set -euo pipefail

# Run ALL current manuscript training settings:
#   A) Tasks 2/4/6 under GKF3/GKF5/GKF7/GKF10/LOSO/LOCO
#   B) Task 2/4 LOSO sensor experiments
#
# Usage:
#   bash 06_run_all_training.sh SOURCE_DATA_ROOT LABEL_XLSX [OUT_ROOT]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 SOURCE_DATA_ROOT LABEL_XLSX [OUT_ROOT]"
  exit 2
fi

SOURCE_DATA_ROOT=$(realpath "$1")
LABEL_XLSX=$(realpath "$2")
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_ROOT=${3:-"$REPO_ROOT/outputs"}

MAIN_OUT="$OUT_ROOT/main_protocols"
SENSOR_OUT="$OUT_ROOT/sensor_loso"

EPOCHS=${EPOCHS:-100}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-20}
JOINT_EPOCHS=${JOINT_EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-8}
SEEDS=${SEEDS:-"42 43 44"}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-1}
SAVE_MODEL=${SAVE_MODEL:-1}
RUN_SENSOR=${RUN_SENSOR:-1}

mkdir -p "$OUT_ROOT"

printf '%s\n' "===== FULL TRAINING PLAN ====="
echo "SOURCE_DATA_ROOT=$SOURCE_DATA_ROOT"
echo "LABEL_XLSX=$LABEL_XLSX"
echo "REPO_ROOT=$REPO_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "MAIN_OUT=$MAIN_OUT"
echo "SENSOR_OUT=$SENSOR_OUT"
echo "Protocols=gkf3 gkf5 gkf7 gkf10 loso loco"
echo "Tasks=2 4 6"
echo "Seeds=$SEEDS"
echo "Epochs=$EPOCHS (Pretrain=$PRETRAIN_EPOCHS Joint=$JOINT_EPOCHS) Batch=$BATCH_SIZE Workers=$NUM_WORKERS AMP=$AMP SaveModel=$SAVE_MODEL"

EPOCHS="$EPOCHS" \
PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS" \
JOINT_EPOCHS="$JOINT_EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
SEEDS="$SEEDS" \
NUM_WORKERS="$NUM_WORKERS" \
AMP="$AMP" \
SAVE_MODEL="$SAVE_MODEL" \
TASKS="2 4 6" \
BALANCED_LOSS_TASKS="" \
PROTOCOLS="gkf3 gkf5 gkf7 gkf10 loso loco" \
LOCO_TRAIN_ONLY_CENTERS="青海" \
bash "$REPO_ROOT/05_run_all_protocols.sh" \
  "$SOURCE_DATA_ROOT" \
  "$LABEL_XLSX" \
  "$REPO_ROOT" \
  "$MAIN_OUT"

if [[ "$RUN_SENSOR" == "1" ]]; then
  echo "===== START LOSO SENSOR EXPERIMENTS ====="
  SEEDS="$SEEDS" \
  NUM_WORKERS="$NUM_WORKERS" \
  AMP="$AMP" \
  UNIAXIAL_DEFINITION=image \
  bash "$REPO_ROOT/sensor_experiments/03_run_sensor_experiments_loso.sh" \
    "$MAIN_OUT" \
    "$REPO_ROOT" \
    "$REPO_ROOT" \
    "$SENSOR_OUT"
fi

echo "[DONE] All training settings completed under: $OUT_ROOT"
