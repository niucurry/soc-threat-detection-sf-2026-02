#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
if [[ $# -lt 2 || $# -gt 3 || ( "$1" != study && "$1" != refit ) ]]; then
  echo "Usage: bash scripts/run_full_label_refit.sh study|refit DATA_DIR [OUTPUT_DIR]"
  exit 2
fi
OUT="${3:-artifacts/v5_2_exp03_full_label}"
mkdir -p "${OUT}"
exec 9>"${OUT}/workflow.lock"
flock -n 9 || { echo "This workflow is already running; do not launch twice"; exit 2; }
ARGS=(--stage "$1" --data-dir "$2" --output-dir "${OUT}")
if [[ "$1" == study ]]; then
  ARGS+=(--device "${REFIT_DEVICE:-auto}" --seeds "${REFIT_SEEDS:-20261015,20261016,20261017}"
    --epochs "${REFIT_EPOCHS:-10}" --patience "${REFIT_PATIENCE:-4}"
    --batch-size "${REFIT_BATCH_SIZE:-2048}" --valid-batch-size "${REFIT_VALID_BATCH_SIZE:-4096}"
    --threads "${REFIT_THREADS:-4}" --num-workers "${REFIT_NUM_WORKERS:-0}"
    --holdout-fraction "${REFIT_HOLDOUT_FRACTION:-0.2}" --split-seed "${REFIT_SPLIT_SEED:-20261015}")
elif [[ "${REFIT_ALLOW_NONIMPROVING:-0}" == 1 ]]; then
  ARGS+=(--allow-nonimproving)
fi
python -u src/run_full_label_refit.py "${ARGS[@]}"
