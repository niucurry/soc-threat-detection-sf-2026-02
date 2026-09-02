#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V8_OUTPUT_ROOT:-artifacts/v8_multiview_evidence}"
PROCESSED_DIR="${V8_PROCESSED_ROOT:-data/processed}"
V6_FEATURE_DIR="${PROCESSED_DIR}/v6"
V8_FEATURE_DIR="${PROCESSED_DIR}/v8"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${V6_FEATURE_DIR}" "${V8_FEATURE_DIR}"

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

V6_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${PROCESSED_DIR}"
  --output-dir "${V6_FEATURE_DIR}"
  --batch-size "${V8_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V8_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V8_HASH_BUCKETS:-65536}"
  --max-tokens "${V8_RAW_MAX_TOKENS:-96}"
)
if [[ "${V8_FORCE_V6_PREPARE:-0}" == "1" ]]; then
  V6_PREPARE_ARGS+=(--force)
fi
python -u src/prepare_v6_content.py "${V6_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_v6_console.log"

V4_PREDICTIONS="${V8_V4_PREDICTIONS:-artifacts/v4_drain_neural/base/valid_predictions.parquet}"
V5_PREDICTIONS="${V8_V5_PREDICTIONS:-artifacts/v5_structured_neural/base/valid_predictions.parquet}"
V6_PREDICTIONS="${V8_V6_PREDICTIONS:-artifacts/v6_content_neural/e2_fusion_raw/valid_predictions.parquet}"
V7_PREDICTIONS="${V8_V7_PREDICTIONS:-artifacts/v7_hierarchical_content/h2_hierarchical_novelty/valid_predictions.parquet}"

train_and_audit() {
  local run_name="$1"
  local content_input="$2"
  local train_path="$3"
  local valid_path="$4"
  local evidence_weight="$5"
  local feature_path="$6"
  local run_coverage_audit="$7"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  if [[ -f "${run_dir}/metrics.json" && "${V8_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} metrics already exist; skipping completed experiment."
  else
    python -u src/train_hierarchical_content.py \
      --experiment-version v8 \
      --novelty-gate count \
      --content-input "${content_input}" \
      --content-view-count 4 \
      --content-tokens-per-view "${V8_TOKENS_PER_VIEW:-64}" \
      --evidence-preservation-weight "${evidence_weight}" \
      --positive-threat-margin "${V8_POSITIVE_THREAT_MARGIN:-0.0}" \
      --allowed-branch-logit-gap "${V8_ALLOWED_BRANCH_LOGIT_GAP:-0.5}" \
      --train "${train_path}" \
      --valid "${valid_path}" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V8_EPOCHS:-10}" \
      --batch-size "${V8_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V8_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V8_LEARNING_RATE:-0.0007}" \
      --threat-class-weight-power "${V8_THREAT_CLASS_WEIGHT_POWER:-0.25}" \
      --subtype-class-weight-power "${V8_SUBTYPE_CLASS_WEIGHT_POWER:-0.0}" \
      --subtype-loss-weight "${V8_SUBTYPE_LOSS_WEIGHT:-0.75}" \
      --metadata-threat-aux-weight "${V8_METADATA_THREAT_AUX_WEIGHT:-0.15}" \
      --content-threat-aux-weight "${V8_CONTENT_THREAT_AUX_WEIGHT:-0.25}" \
      --metadata-subtype-aux-weight "${V8_METADATA_SUBTYPE_AUX_WEIGHT:-0.35}" \
      --threat-threshold "${V8_THREAT_THRESHOLD:-0.5}" \
      --novelty-pseudocount "${V8_NOVELTY_PSEUDOCOUNT:-32}" \
      --patience "${V8_PATIENCE:-4}" \
      --num-workers "${V8_NUM_WORKERS:-4}" \
      --hash-buckets "${V8_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V8_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V8_CONTENT_OUTPUT_DIM:-128}" \
      --token-dropout "${V8_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V8_CATEGORY_DROPOUT:-0.02}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  local audit_args=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${feature_path}"
    --output-dir "${run_dir}/analysis"
  )
  [[ -f "${V4_PREDICTIONS}" ]] && audit_args+=(--v4-predictions "${V4_PREDICTIONS}")
  [[ -f "${V5_PREDICTIONS}" ]] && audit_args+=(--v5-predictions "${V5_PREDICTIONS}")
  [[ -f "${V6_PREDICTIONS}" ]] && audit_args+=(--v6-predictions "${V6_PREDICTIONS}")
  [[ -f "${V7_PREDICTIONS}" ]] && audit_args+=(--v7-predictions "${V7_PREDICTIONS}")
  python src/analyze_v7_errors.py "${audit_args[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"

  if [[ "${run_coverage_audit}" == "1" ]]; then
    python src/analyze_v8_coverage.py \
      --predictions "${run_dir}/valid_predictions.parquet" \
      --features "${feature_path}" \
      --tokens-per-view "${V8_TOKENS_PER_VIEW:-64}" \
      --output "${run_dir}/analysis/content_coverage.json" \
      2>&1 | tee "${run_dir}/coverage_console.log"
  fi
}

# B runs first because it reuses V6 files and can produce a useful result before
# the more expensive V8 multi-view preparation has finished.
train_and_audit \
  b1_raw_evidence \
  raw \
  "${V6_FEATURE_DIR}/v6_train.parquet" \
  "${V6_FEATURE_DIR}/v6_valid.parquet" \
  "${V8_EVIDENCE_WEIGHT:-0.20}" \
  "${V6_FEATURE_DIR}/v6_valid.parquet" \
  0

V8_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --v6-feature-dir "${V6_FEATURE_DIR}"
  --output-dir "${V8_FEATURE_DIR}"
  --batch-size "${V8_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V8_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V8_HASH_BUCKETS:-65536}"
  --tokens-per-view "${V8_TOKENS_PER_VIEW:-64}"
  --chars-per-view "${V8_CHARS_PER_VIEW:-4096}"
)
if [[ "${V8_FORCE_PREPARE:-0}" == "1" ]]; then
  V8_PREPARE_ARGS+=(--force)
fi
python -u src/prepare_v8_content.py "${V8_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_v8_console.log"

train_and_audit \
  a1_multiview_standard \
  multiview \
  "${V8_FEATURE_DIR}/v8_train.parquet" \
  "${V8_FEATURE_DIR}/v8_valid.parquet" \
  0.0 \
  "${V8_FEATURE_DIR}/v8_valid.parquet" \
  1

train_and_audit \
  c1_multiview_evidence \
  multiview \
  "${V8_FEATURE_DIR}/v8_train.parquet" \
  "${V8_FEATURE_DIR}/v8_valid.parquet" \
  "${V8_EVIDENCE_WEIGHT:-0.20}" \
  "${V8_FEATURE_DIR}/v8_valid.parquet" \
  1

COMPARE_ARGS=(
  --root "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/comparison.json"
)
V4_METRICS="${V8_V4_METRICS:-artifacts/v4_drain_neural/base/metrics.json}"
V5_METRICS="${V8_V5_METRICS:-artifacts/v5_structured_neural/base/metrics.json}"
V6_METRICS="${V8_V6_METRICS:-artifacts/v6_content_neural/e2_fusion_raw/metrics.json}"
V7_METRICS="${V8_V7_METRICS:-artifacts/v7_hierarchical_content/h2_hierarchical_novelty/metrics.json}"
[[ -f "${V4_METRICS}" ]] && COMPARE_ARGS+=(--v4-metrics "${V4_METRICS}")
[[ -f "${V5_METRICS}" ]] && COMPARE_ARGS+=(--v5-metrics "${V5_METRICS}")
[[ -f "${V6_METRICS}" ]] && COMPARE_ARGS+=(--v6-metrics "${V6_METRICS}")
[[ -f "${V7_METRICS}" ]] && COMPARE_ARGS+=(--v7-metrics "${V7_METRICS}")
python src/compare_v7_experiments.py "${COMPARE_ARGS[@]}"

echo "V8 comparison: ${OUTPUT_ROOT}/comparison.json"
echo "Each run contains metrics.json, valid_predictions.parquet, and analysis/error_summary.json."
