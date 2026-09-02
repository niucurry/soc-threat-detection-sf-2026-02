#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V6_OUTPUT_ROOT:-artifacts/v6_content_neural}"
PROCESSED_DIR="${V6_PROCESSED_DIR:-data/processed}"
V6_FEATURE_DIR="${PROCESSED_DIR}/v6"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${V6_FEATURE_DIR}"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt
python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${PROCESSED_DIR}/v1_train.parquet" || ! -f "${PROCESSED_DIR}/v1_valid.parquet" ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir "${PROCESSED_DIR}"
else
  echo "V1 base features already exist; reusing them."
fi

PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${PROCESSED_DIR}"
  --output-dir "${V6_FEATURE_DIR}"
  --batch-size "${V6_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V6_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V6_HASH_BUCKETS:-65536}"
  --max-tokens "${V6_MAX_TOKENS:-96}"
)
if [[ "${V6_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
# Always enter the preparer: it validates joined files and every shard, while
# reusing complete outputs. This makes rerunning the same command after a cloud
# shutdown safe without paying the preprocessing cost again.
python -u src/prepare_v6_content.py "${PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"

MODES=(content fusion_raw fusion_field)
RUN_NAMES=(e1_content_raw e2_fusion_raw e3_fusion_field)

for index in "${!MODES[@]}"; do
  mode="${MODES[$index]}"
  run_name="${RUN_NAMES[$index]}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" && "${V6_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} metrics already exist; skipping completed experiment."
  else
    python -u src/train_content_neural.py \
      --mode "${mode}" \
      --train "${V6_FEATURE_DIR}/v6_train.parquet" \
      --valid "${V6_FEATURE_DIR}/v6_valid.parquet" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V6_EPOCHS:-12}" \
      --batch-size "${V6_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V6_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V6_LEARNING_RATE:-0.001}" \
      --class-weight-power "${V6_CLASS_WEIGHT_POWER:-0.0}" \
      --patience "${V6_PATIENCE:-3}" \
      --num-workers "${V6_NUM_WORKERS:-4}" \
      --hash-buckets "${V6_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V6_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V6_CONTENT_OUTPUT_DIM:-128}" \
      --content-aux-weight "${V6_CONTENT_AUX_WEIGHT:-0.25}" \
      --token-dropout "${V6_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V6_CATEGORY_DROPOUT:-0.05}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  AUDIT_ARGS=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${V6_FEATURE_DIR}/v6_valid.parquet"
    --output-dir "${run_dir}/analysis"
  )
  V4_PREDICTIONS="${V6_V4_PREDICTIONS:-artifacts/v4_drain_neural/base/valid_predictions.parquet}"
  V5_PREDICTIONS="${V6_V5_PREDICTIONS:-artifacts/v5_structured_neural/base/valid_predictions.parquet}"
  if [[ -f "${V4_PREDICTIONS}" ]]; then
    AUDIT_ARGS+=(--v4-predictions "${V4_PREDICTIONS}")
  fi
  if [[ -f "${V5_PREDICTIONS}" ]]; then
    AUDIT_ARGS+=(--v5-predictions "${V5_PREDICTIONS}")
  fi
  python src/analyze_v6_errors.py "${AUDIT_ARGS[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
done

COMPARE_ARGS=(
  --root "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/comparison.json"
)
V4_METRICS="${V6_V4_METRICS:-artifacts/v4_drain_neural/base/metrics.json}"
V5_METRICS="${V6_V5_METRICS:-artifacts/v5_structured_neural/base/metrics.json}"
if [[ -f "${V4_METRICS}" ]]; then
  COMPARE_ARGS+=(--v4-metrics "${V4_METRICS}")
fi
if [[ -f "${V5_METRICS}" ]]; then
  COMPARE_ARGS+=(--v5-metrics "${V5_METRICS}")
fi
python src/compare_v6_experiments.py "${COMPARE_ARGS[@]}"

echo "V6 comparison: ${OUTPUT_ROOT}/comparison.json"
echo "V6 preparation manifest: ${V6_FEATURE_DIR}/v6_manifest.json"
echo "Each run contains metrics.json, valid_predictions.parquet, and analysis/error_summary.json."
