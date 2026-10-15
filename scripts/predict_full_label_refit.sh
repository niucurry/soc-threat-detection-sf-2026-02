#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
if [[ $# != 3 ]]; then
  echo "Usage: bash scripts/predict_full_label_refit.sh TEST_INPUT.parquet MODEL.pt OUTPUT_DIR"
  exit 2
fi
mkdir -p "$3"
exec 9>"$3/predict.lock"
flock -n 9 || { echo "Prediction already running"; exit 2; }
python -u src/predict_submission.py --input "$1" --refit-checkpoint "$2" --output-dir "$3" \
  --device "${PREDICT_DEVICE:-auto}" --chunk-size "${PREDICT_CHUNK_SIZE:-10000}" \
  --batch-size "${PREDICT_BATCH_SIZE:-4096}" --threads "${PREDICT_THREADS:-4}"
