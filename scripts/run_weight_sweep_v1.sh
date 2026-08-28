#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

TRAIN_PATH="${1:-data/processed/v1_train.parquet}"
VALID_PATH="${2:-data/processed/v1_valid.parquet}"
OUTPUT_ROOT="${WEIGHT_SWEEP_OUTPUT:-artifacts/v1_weight_sweep}"

for required_file in "${TRAIN_PATH}" "${VALID_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required feature file does not exist: ${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"

for power in 0.00 0.25 0.50 0.75; do
  run_name="power_${power/./p}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${output_dir}"
  echo "Starting class-weight power ${power}"
  python -u src/train_npu_tabular.py \
    --train "${TRAIN_PATH}" \
    --valid "${VALID_PATH}" \
    --device auto \
    --epochs 5 \
    --batch-size 8192 \
    --learning-rate 0.002 \
    --class-weight-power "${power}" \
    --patience 2 \
    --num-workers 4 \
    --max-train-rows 200000 \
    --max-valid-rows 200000 \
    --output-dir "${output_dir}" \
    2>&1 | tee "${output_dir}/train_console.log"
done

python src/summarize_experiments.py \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/comparison.json"
