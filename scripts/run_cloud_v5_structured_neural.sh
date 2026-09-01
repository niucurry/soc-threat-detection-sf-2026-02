#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V5_OUTPUT_ROOT:-artifacts/v5_structured_neural}"
PROCESSED_DIR="${V5_PROCESSED_DIR:-data/processed}"
V5_FEATURE_DIR="${PROCESSED_DIR}/v5"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/base" "${V5_FEATURE_DIR}"

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
  --output-dir "${V5_FEATURE_DIR}"
  --model-dir "${OUTPUT_ROOT}/template_model"
)

if [[ "${V5_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
elif [[ -f "${V5_FEATURE_DIR}/v5_train.parquet" && -f "${V5_FEATURE_DIR}/v5_valid.parquet" ]]; then
  echo "V5.1 feature files already exist; skipping structured parsing and Drain preparation."
  PREPARE_ARGS=()
fi

if [[ ${#PREPARE_ARGS[@]} -gt 0 ]]; then
  python -u src/prepare_v5_features.py "${PREPARE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"
fi

python -u src/train_npu_tabular.py \
  --feature-set v5 \
  --train "${V5_FEATURE_DIR}/v5_train.parquet" \
  --valid "${V5_FEATURE_DIR}/v5_valid.parquet" \
  --device auto \
  --epochs "${V5_EPOCHS:-20}" \
  --batch-size "${V5_BATCH_SIZE:-8192}" \
  --learning-rate "${V5_LEARNING_RATE:-0.002}" \
  --class-weight-power "${V5_CLASS_WEIGHT_POWER:-0.0}" \
  --patience "${V5_PATIENCE:-4}" \
  --num-workers "${V5_NUM_WORKERS:-4}" \
  --output-dir "${OUTPUT_ROOT}/base" \
  2>&1 | tee "${OUTPUT_ROOT}/base/train_console.log"

AUDIT_ARGS=(
  --predictions "${OUTPUT_ROOT}/base/valid_predictions.parquet"
  --features "${V5_FEATURE_DIR}/v5_valid.parquet"
  --output-dir "${OUTPUT_ROOT}/analysis"
)
V4_PREDICTIONS="${V5_V4_PREDICTIONS:-artifacts/v4_drain_neural/base/valid_predictions.parquet}"
if [[ -f "${V4_PREDICTIONS}" ]]; then
  AUDIT_ARGS+=(--v4-predictions "${V4_PREDICTIONS}")
fi
python src/analyze_v5_errors.py "${AUDIT_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/analysis_console.log"

python src/summarize_experiments.py \
  --root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/comparison.json"

echo "V5.1 metrics: ${OUTPUT_ROOT}/base/metrics.json"
echo "V5.1 predictions: ${OUTPUT_ROOT}/base/valid_predictions.parquet"
echo "V5.1 feature audit: ${V5_FEATURE_DIR}/v5_manifest.json"
echo "V5.1 template manifest: ${OUTPUT_ROOT}/template_model/manifest.json"
echo "V5.1 error audit: ${OUTPUT_ROOT}/analysis/error_summary.json"
