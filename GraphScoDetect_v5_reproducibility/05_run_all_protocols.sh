#!/usr/bin/env bash
set -euo pipefail

# GraphScoDetect v5 full protocol runner.
# Default sequence:
#   gkf3 -> gkf5 -> gkf7 -> gkf10 -> loso -> loco
#
# Usage:
#   bash 05_run_all_protocols.sh \
#     /path/to/QC_SOURCE_DATA_ROOT \
#     /path/to/label_v5.xlsx \
#     /path/to/GraphScoDetect_v5_reproducibility \
#     /path/to/output/main_protocols

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 SOURCE_DATA_ROOT LABEL_XLSX PROJECT_ROOT [OUT_BASE]"
  exit 2
fi

SOURCE_DATA_ROOT=$(realpath "$1")
LABEL_XLSX=$(realpath "$2")
PROJECT_ROOT=$(realpath "$3")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_BASE=${4:-"$SCRIPT_DIR/outputs/main_protocols"}

DATA_ROOT="$OUT_BASE/data"
SPLIT_ROOT="$OUT_BASE/splits"
RESULT_ROOT="$OUT_BASE/results"
SUMMARY_ROOT="$OUT_BASE/summary"
LOG_ROOT="$OUT_BASE/logs"
mkdir -p "$DATA_ROOT" "$SPLIT_ROOT" "$RESULT_ROOT" "$SUMMARY_ROOT" "$LOG_ROOT"

EPOCHS=${EPOCHS:-100}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-20}
JOINT_EPOCHS=${JOINT_EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-8}
SEEDS_STR=${SEEDS:-"42 43 44"}
TASKS_STR=${TASKS:-"2 4 6"}
BALANCED_STR=${BALANCED_LOSS_TASKS:-""}
PROTOCOLS_STR=${PROTOCOLS:-"gkf3 gkf5 gkf7 gkf10 loso loco"}
LOCO_TRAIN_ONLY_CENTERS_STR=${LOCO_TRAIN_ONLY_CENTERS:-"青海"}
LOCATION_SCOPE=${LOCATION_SCOPE:-primary_all}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-4}
AMP=${AMP:-1}
SAVE_MODEL=${SAVE_MODEL:-0}
OVERWRITE=${OVERWRITE:-0}
SPLIT_SEED=${SPLIT_SEED:-42}
LIMIT_FOLDS=${LIMIT_FOLDS:-0}

read -r -a SEED_ARGS <<< "$SEEDS_STR"
read -r -a TASK_ARGS <<< "$TASKS_STR"
read -r -a BALANCED_ARGS <<< "$BALANCED_STR"
read -r -a PROTOCOL_ARGS <<< "$PROTOCOLS_STR"
read -r -a LOCO_TRAIN_ONLY_CENTER_ARGS <<< "$LOCO_TRAIN_ONLY_CENTERS_STR"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

printf '%s\n' "===== CONFIGURATION ====="
echo "SOURCE_DATA_ROOT=$SOURCE_DATA_ROOT"
echo "LABEL_XLSX=$LABEL_XLSX"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "OUT_BASE=$OUT_BASE"
echo "TASKS=${TASK_ARGS[*]}"
echo "PROTOCOL_ORDER=${PROTOCOL_ARGS[*]}"
echo "SEEDS=${SEED_ARGS[*]}"
echo "EPOCHS=$EPOCHS (PRETRAIN=$PRETRAIN_EPOCHS, JOINT=$JOINT_EPOCHS)"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "BALANCED_LOSS_TASKS=${BALANCED_ARGS[*]:-(none)}"
echo "LOCO_TRAIN_ONLY_CENTERS=${LOCO_TRAIN_ONLY_CENTER_ARGS[*]}"
echo "EXCLUSION_NOTE_KEYWORD=不要数据"
echo "DEVICE=$DEVICE AMP=$AMP NUM_WORKERS=$NUM_WORKERS"

python "$SCRIPT_DIR/00_validate_inputs.py" \
  "$SOURCE_DATA_ROOT" "$LABEL_XLSX" \
  --exclude_note_keywords "不要数据" \
  2>&1 | tee "$LOG_ROOT/00_validate_inputs.log"

python "$SCRIPT_DIR/01_prepare_tasks.py" \
  --label_excel "$LABEL_XLSX" \
  --label_sheet auto \
  --source_data_root "$SOURCE_DATA_ROOT" \
  --out_data_root "$DATA_ROOT" \
  --location_scope "$LOCATION_SCOPE" \
  --exclude_note_keywords "不要数据" \
  --copy_label_excel \
  2>&1 | tee "$LOG_ROOT/01_prepare_tasks.log"

python "$SCRIPT_DIR/02_make_all_subject_splits.py" \
  --data_root "$DATA_ROOT" \
  --out_split_root "$SPLIT_ROOT" \
  --tasks "${TASK_ARGS[@]}" \
  --protocols "${PROTOCOL_ARGS[@]}" \
  --seed "$SPLIT_SEED" \
  --loco_train_only_centers "${LOCO_TRAIN_ONLY_CENTER_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/02_make_splits.log"

for PROTOCOL in "${PROTOCOL_ARGS[@]}"; do
  echo "===== START $PROTOCOL ====="
  TRAIN_ARGS=(
    --data_root "$DATA_ROOT"
    --split_root "$SPLIT_ROOT"
    --out_root "$RESULT_ROOT"
    --project_root "$PROJECT_ROOT"
    --tasks "${TASK_ARGS[@]}"
    --protocols "$PROTOCOL"
    --seeds "${SEED_ARGS[@]}"
    --epochs "$EPOCHS"
    --pretrain_epochs "$PRETRAIN_EPOCHS"
    --joint_epochs "$JOINT_EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr 1e-4
    --weight_decay 1e-4
    --dropout 0.2
    --resample_len 500
    --segment_len 25
    --hidden_dim 64
    --lstm_hidden 128
    --lambda_inter 0.5
    --gamma_class 1.0
    --intra_margin 1.0
    --temperature 0.1
    --device "$DEVICE"
    --num_workers "$NUM_WORKERS"
  )
  if [[ ${#BALANCED_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS+=(--balanced_loss_tasks "${BALANCED_ARGS[@]}")
  else
    TRAIN_ARGS+=(--balanced_loss_tasks)
  fi
  if [[ "$AMP" == "1" ]]; then TRAIN_ARGS+=(--amp); fi
  if [[ "$SAVE_MODEL" == "1" ]]; then TRAIN_ARGS+=(--save_model); fi
  if [[ "$OVERWRITE" == "1" ]]; then TRAIN_ARGS+=(--overwrite); fi
  if [[ "$LIMIT_FOLDS" -gt 0 ]]; then TRAIN_ARGS+=(--limit_folds "$LIMIT_FOLDS"); fi

  python "$SCRIPT_DIR/03_train_all_protocols.py" "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$LOG_ROOT/03_train_${PROTOCOL}.log"
  echo "===== FINISH $PROTOCOL ====="
done

python "$SCRIPT_DIR/04_summarize_results.py" \
  --result_root "$RESULT_ROOT" \
  --out_dir "$SUMMARY_ROOT" \
  --out_xlsx graphscodetect_v5_protocol_summary.xlsx \
  --loco_train_only_centers "${LOCO_TRAIN_ONLY_CENTER_ARGS[@]}" \
  2>&1 | tee "$LOG_ROOT/04_summarize.log"

echo "[DONE] All outputs: $OUT_BASE"
echo "[MAIN WORKBOOK] $SUMMARY_ROOT/graphscodetect_v5_protocol_summary.xlsx"
echo "[MAIN METRICS] $SUMMARY_ROOT/01_main_subject_ensemble_metrics.csv"
echo "[LOCO PER CENTER] $SUMMARY_ROOT/14_loco_per_center_ensemble_metrics.csv"
