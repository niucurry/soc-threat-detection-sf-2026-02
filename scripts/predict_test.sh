#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/predict_test.sh TEST_INPUT.parquet BACKUP.tar.gz [OUTPUT_DIR]"
  exit 2
fi
OUTPUT_DIR="${3:-artifacts/submission_v5_2}"
mkdir -p "${OUTPUT_DIR}"
# Prevent two cloud processes from writing the same shards simultaneously.
exec 9>"${OUTPUT_DIR}/predict.lock"
flock -n 9 || { echo "Prediction is already running for this output directory"; exit 2; }
python -u src/predict_submission.py \
  --input "$1" --archive "$2" --output-dir "${OUTPUT_DIR}" \
  --device "${PREDICT_DEVICE:-auto}" --chunk-size "${PREDICT_CHUNK_SIZE:-10000}" \
  --batch-size "${PREDICT_BATCH_SIZE:-4096}" --threads "${PREDICT_THREADS:-4}"
