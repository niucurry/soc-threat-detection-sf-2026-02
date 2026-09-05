#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V1_0_OUTPUT_ROOT:-artifacts/v1_0_tabular}"
PROCESSED_DIR="${V1_0_PROCESSED_ROOT:-data/processed}"
FEATURE_DIR="${PROCESSED_DIR}/v1_0"
for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${PROCESSED_DIR}"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${FEATURE_DIR}/tabular_train.parquet" || ! -f "${FEATURE_DIR}/tabular_valid.parquet" ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${FEATURE_DIR}"
else
  echo "Prepared feature files already exist; skipping feature preparation."
fi

python -u src/train_npu_tabular.py \
  --feature-set tabular \
  --train "${FEATURE_DIR}/tabular_train.parquet" \
  --valid "${FEATURE_DIR}/tabular_valid.parquet" \
  --device auto \
  --epochs "${V1_0_EPOCHS:-20}" \
  --batch-size "${V1_0_BATCH_SIZE:-8192}" \
  --learning-rate "${V1_0_LEARNING_RATE:-0.002}" \
  --class-weight-power "${V1_0_CLASS_WEIGHT_POWER:-0.0}" \
  --patience "${V1_0_PATIENCE:-4}" \
  --num-workers "${V1_0_NUM_WORKERS:-4}" \
  --output-dir "${OUTPUT_ROOT}" \
  2>&1 | tee "${OUTPUT_ROOT}/train_console.log"
