#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:-artifacts/v4_drain_neural}"
PROCESSED_DIR="${V4_PROCESSED_DIR:-data/processed}"
V4_FEATURE_DIR="${PROCESSED_DIR}/v4"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/base" "${V4_FEATURE_DIR}"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${PROCESSED_DIR}/v1_train.parquet" || ! -f "${PROCESSED_DIR}/v1_valid.parquet" ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${PROCESSED_DIR}"
else
  echo "V1 base feature files already exist; reusing them."
fi

PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${PROCESSED_DIR}"
  --output-dir "${V4_FEATURE_DIR}"
  --model-dir "${OUTPUT_ROOT}/template_model"
)

if [[ "${V4_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
elif [[ -f "${V4_FEATURE_DIR}/v4_train.parquet" && -f "${V4_FEATURE_DIR}/v4_valid.parquet" ]]; then
  echo "V4 feature files already exist; skipping hybrid parser and Drain preparation."
  PREPARE_ARGS=()
fi

if [[ ${#PREPARE_ARGS[@]} -gt 0 ]]; then
  python -u src/prepare_v4_features.py "${PREPARE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"
fi

python -u src/train_npu_tabular.py \
  --feature-set v4 \
  --train "${V4_FEATURE_DIR}/v4_train.parquet" \
  --valid "${V4_FEATURE_DIR}/v4_valid.parquet" \
  --device auto \
  --epochs "${V4_EPOCHS:-20}" \
  --batch-size "${V4_BATCH_SIZE:-8192}" \
  --learning-rate "${V4_LEARNING_RATE:-0.002}" \
  --class-weight-power "${V4_CLASS_WEIGHT_POWER:-0.0}" \
  --patience "${V4_PATIENCE:-4}" \
  --num-workers "${V4_NUM_WORKERS:-4}" \
  --output-dir "${OUTPUT_ROOT}/base" \
  2>&1 | tee "${OUTPUT_ROOT}/base/train_console.log"

python src/summarize_experiments.py \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/comparison.json"

echo "V4 metrics: ${OUTPUT_ROOT}/base/metrics.json"
echo "V4 predictions: ${OUTPUT_ROOT}/base/valid_predictions.parquet"
echo "Template manifest: ${OUTPUT_ROOT}/template_model/manifest.json"
