#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V1_1_OUTPUT_ROOT:-artifacts/v1_1_drain}"
PROCESSED_DIR="${V1_1_PROCESSED_ROOT:-data/processed}"
TABULAR_FEATURE_DIR="${PROCESSED_DIR}/v1_0"
FEATURE_DIR="${PROCESSED_DIR}/v1_1"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/base" "${FEATURE_DIR}"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${TABULAR_FEATURE_DIR}/tabular_train.parquet" || ! -f "${TABULAR_FEATURE_DIR}/tabular_valid.parquet" ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${TABULAR_FEATURE_DIR}"
else
  echo "v1.0 base feature files already exist; reusing them."
fi

PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${TABULAR_FEATURE_DIR}"
  --output-dir "${FEATURE_DIR}"
  --model-dir "${OUTPUT_ROOT}/template_model"
)

if [[ "${V1_1_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
elif [[ -f "${FEATURE_DIR}/drain_train.parquet" && -f "${FEATURE_DIR}/drain_valid.parquet" ]]; then
  echo "v1.1 feature files already exist; skipping grouped-Drain preparation."
  PREPARE_ARGS=()
fi

if [[ ${#PREPARE_ARGS[@]} -gt 0 ]]; then
  python -u src/prepare_drain_features.py "${PREPARE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"
fi

python -u src/train_npu_tabular.py \
  --feature-set drain \
  --train "${FEATURE_DIR}/drain_train.parquet" \
  --valid "${FEATURE_DIR}/drain_valid.parquet" \
  --device auto \
  --epochs "${V1_1_EPOCHS:-20}" \
  --batch-size "${V1_1_BATCH_SIZE:-8192}" \
  --learning-rate "${V1_1_LEARNING_RATE:-0.002}" \
  --class-weight-power "${V1_1_CLASS_WEIGHT_POWER:-0.0}" \
  --patience "${V1_1_PATIENCE:-4}" \
  --num-workers "${V1_1_NUM_WORKERS:-4}" \
  --output-dir "${OUTPUT_ROOT}/base" \
  2>&1 | tee "${OUTPUT_ROOT}/base/train_console.log"

python src/compare_experiments.py \
  --scan-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/comparison.json"

echo "v1.1 metrics: ${OUTPUT_ROOT}/base/metrics.json"
echo "v1.1 predictions: ${OUTPUT_ROOT}/base/valid_predictions.parquet"
echo "Template manifest: ${OUTPUT_ROOT}/template_model/manifest.json"
