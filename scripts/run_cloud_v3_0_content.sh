#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V3_0_OUTPUT_ROOT:-artifacts/v3_0_content}"
PROCESSED_DIR="${V3_0_PROCESSED_ROOT:-data/processed}"
FEATURE_DIR="${PROCESSED_DIR}/v3_0"
TABULAR_FEATURE_DIR="${PROCESSED_DIR}/v1_0"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${FEATURE_DIR}"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt
python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${TABULAR_FEATURE_DIR}/tabular_train.parquet" || ! -f "${TABULAR_FEATURE_DIR}/tabular_valid.parquet" ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${TABULAR_FEATURE_DIR}"
else
  echo "v1.0 base features already exist; reusing them."
fi

PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${TABULAR_FEATURE_DIR}"
  --output-dir "${FEATURE_DIR}"
  --batch-size "${V3_0_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V3_0_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V3_0_HASH_BUCKETS:-65536}"
  --max-tokens "${V3_0_MAX_TOKENS:-96}"
)
if [[ "${V3_0_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
# Always enter the preparer: it validates joined files and every shard, while
# reusing complete outputs. This makes rerunning the same command after a cloud
# shutdown safe without paying the preprocessing cost again.
python -u src/prepare_content_features.py "${PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"

MODES=(content fusion_raw fusion_field)
RUN_NAMES=(exp01_content_only exp02_raw_fusion exp03_field_fusion)

for index in "${!MODES[@]}"; do
  mode="${MODES[$index]}"
  run_name="${RUN_NAMES[$index]}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" && "${V3_0_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} metrics already exist; skipping completed experiment."
  else
    python -u src/train_content_neural.py \
      --mode "${mode}" \
      --train "${FEATURE_DIR}/content_train.parquet" \
      --valid "${FEATURE_DIR}/content_valid.parquet" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V3_0_EPOCHS:-12}" \
      --batch-size "${V3_0_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V3_0_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V3_0_LEARNING_RATE:-0.001}" \
      --class-weight-power "${V3_0_CLASS_WEIGHT_POWER:-0.0}" \
      --patience "${V3_0_PATIENCE:-3}" \
      --num-workers "${V3_0_NUM_WORKERS:-4}" \
      --hash-buckets "${V3_0_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V3_0_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V3_0_CONTENT_OUTPUT_DIM:-128}" \
      --content-aux-weight "${V3_0_CONTENT_AUX_WEIGHT:-0.25}" \
      --token-dropout "${V3_0_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V3_0_CATEGORY_DROPOUT:-0.05}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  AUDIT_ARGS=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${FEATURE_DIR}/content_valid.parquet"
    --output-dir "${run_dir}/analysis"
  )
  DRAIN_PREDICTIONS="${V3_0_DRAIN_PREDICTIONS:-artifacts/v1_1_drain/base/valid_predictions.parquet}"
  STRUCTURED_PREDICTIONS="${V3_0_STRUCTURED_PREDICTIONS:-artifacts/v1_2_structured/base/valid_predictions.parquet}"
  if [[ -f "${DRAIN_PREDICTIONS}" ]]; then
    AUDIT_ARGS+=(--drain-predictions "${DRAIN_PREDICTIONS}")
  fi
  if [[ -f "${STRUCTURED_PREDICTIONS}" ]]; then
    AUDIT_ARGS+=(--structured-predictions "${STRUCTURED_PREDICTIONS}")
  fi
  python src/analyze_content_errors.py "${AUDIT_ARGS[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
done

COMPARE_ARGS=(
  --scan-root "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/comparison.json"
)
DRAIN_METRICS="${V3_0_DRAIN_METRICS:-artifacts/v1_1_drain/base/metrics.json}"
STRUCTURED_METRICS="${V3_0_STRUCTURED_METRICS:-artifacts/v1_2_structured/base/metrics.json}"
if [[ -f "${DRAIN_METRICS}" ]]; then
  COMPARE_ARGS+=(--run "v1.1=${DRAIN_METRICS}")
fi
if [[ -f "${STRUCTURED_METRICS}" ]]; then
  COMPARE_ARGS+=(--run "v1.2=${STRUCTURED_METRICS}")
fi
python src/compare_experiments.py "${COMPARE_ARGS[@]}"

echo "v3.0 comparison: ${OUTPUT_ROOT}/comparison.json"
echo "v3.0 preparation manifest: ${FEATURE_DIR}/content_manifest.json"
echo "Each run contains metrics.json, valid_predictions.parquet, and analysis/error_summary.json."
