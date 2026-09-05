#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V4_1_OUTPUT_ROOT:-artifacts/v4_1_multiview}"
PROCESSED_DIR="${V4_1_PROCESSED_ROOT:-data/processed}"
CONTENT_FEATURE_DIR="${PROCESSED_DIR}/v3_0"
MULTIVIEW_FEATURE_DIR="${PROCESSED_DIR}/v4_1"
TABULAR_FEATURE_DIR="${PROCESSED_DIR}/v1_0"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CONTENT_FEATURE_DIR}" "${MULTIVIEW_FEATURE_DIR}"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt
python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f "${TABULAR_FEATURE_DIR}/tabular_train.parquet" || ! -f "${TABULAR_FEATURE_DIR}/tabular_valid.parquet" ]]; then
  python src/prepare_features.py --data-dir "${RAW_DATA_DIR}" --output-dir "${TABULAR_FEATURE_DIR}"
fi

CONTENT_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${TABULAR_FEATURE_DIR}"
  --output-dir "${CONTENT_FEATURE_DIR}"
  --batch-size "${V4_1_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V4_1_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V4_1_HASH_BUCKETS:-65536}"
  --max-tokens "${V4_1_RAW_MAX_TOKENS:-96}"
)
[[ "${V4_1_FORCE_CONTENT_PREPARE:-0}" == "1" ]] && CONTENT_PREPARE_ARGS+=(--force)
python -u src/prepare_content_features.py "${CONTENT_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_content_console.log"

DRAIN_PREDICTIONS="${V4_1_DRAIN_PREDICTIONS:-artifacts/v1_1_drain/base/valid_predictions.parquet}"
STRUCTURED_PREDICTIONS="${V4_1_STRUCTURED_PREDICTIONS:-artifacts/v1_2_structured/base/valid_predictions.parquet}"
CONTENT_PREDICTIONS="${V4_1_CONTENT_PREDICTIONS:-artifacts/v3_0_content/exp02_raw_fusion/valid_predictions.parquet}"
HIERARCHICAL_PREDICTIONS="${V4_1_HIERARCHICAL_PREDICTIONS:-artifacts/v4_0_hierarchical/exp02_novelty/valid_predictions.parquet}"

train_and_audit() {
  local run_name="$1"
  local content_input="$2"
  local train_path="$3"
  local valid_path="$4"
  local evidence_weight="$5"
  local run_coverage_audit="$6"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  if [[ -s "${run_dir}/metrics.json" \
        && -s "${run_dir}/model.pt" \
        && -s "${run_dir}/valid_predictions.parquet" \
        && "${V4_1_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} is complete; reusing it."
  else
    python -u src/train_hierarchical_content.py \
      --experiment-version v4.1 \
      --novelty-gate count \
      --content-input "${content_input}" \
      --content-view-count 4 \
      --content-tokens-per-view "${V4_1_TOKENS_PER_VIEW:-64}" \
      --evidence-preservation-weight "${evidence_weight}" \
      --positive-threat-margin "${V4_1_POSITIVE_THREAT_MARGIN:-0.0}" \
      --allowed-branch-logit-gap "${V4_1_ALLOWED_BRANCH_LOGIT_GAP:-0.5}" \
      --train "${train_path}" \
      --valid "${valid_path}" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V4_1_EPOCHS:-10}" \
      --batch-size "${V4_1_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V4_1_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V4_1_LEARNING_RATE:-0.0007}" \
      --threat-class-weight-power "${V4_1_THREAT_CLASS_WEIGHT_POWER:-0.25}" \
      --subtype-class-weight-power "${V4_1_SUBTYPE_CLASS_WEIGHT_POWER:-0.0}" \
      --subtype-loss-weight "${V4_1_SUBTYPE_LOSS_WEIGHT:-0.75}" \
      --metadata-threat-aux-weight "${V4_1_METADATA_THREAT_AUX_WEIGHT:-0.15}" \
      --content-threat-aux-weight "${V4_1_CONTENT_THREAT_AUX_WEIGHT:-0.25}" \
      --metadata-subtype-aux-weight "${V4_1_METADATA_SUBTYPE_AUX_WEIGHT:-0.35}" \
      --threat-threshold "${V4_1_THREAT_THRESHOLD:-0.5}" \
      --novelty-pseudocount "${V4_1_NOVELTY_PSEUDOCOUNT:-32}" \
      --patience "${V4_1_PATIENCE:-4}" \
      --num-workers "${V4_1_NUM_WORKERS:-4}" \
      --hash-buckets "${V4_1_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V4_1_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V4_1_CONTENT_OUTPUT_DIM:-128}" \
      --token-dropout "${V4_1_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V4_1_CATEGORY_DROPOUT:-0.02}" \
      --seed "${V4_1_SEED:-20260828}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  local audit_args=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${valid_path}"
    --output-dir "${run_dir}/analysis"
  )
  [[ -f "${DRAIN_PREDICTIONS}" ]] && audit_args+=(--drain-predictions "${DRAIN_PREDICTIONS}")
  [[ -f "${STRUCTURED_PREDICTIONS}" ]] && audit_args+=(--structured-predictions "${STRUCTURED_PREDICTIONS}")
  [[ -f "${CONTENT_PREDICTIONS}" ]] && audit_args+=(--content-predictions "${CONTENT_PREDICTIONS}")
  [[ -f "${HIERARCHICAL_PREDICTIONS}" ]] && audit_args+=(--hierarchical-predictions "${HIERARCHICAL_PREDICTIONS}")
  python src/analyze_hierarchical_errors.py "${audit_args[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"

  if [[ "${run_coverage_audit}" == "1" ]]; then
    python src/analyze_multiview_coverage.py \
      --predictions "${run_dir}/valid_predictions.parquet" \
      --features "${valid_path}" \
      --tokens-per-view "${V4_1_TOKENS_PER_VIEW:-64}" \
      --output "${run_dir}/analysis/content_coverage.json" \
      2>&1 | tee "${run_dir}/coverage_console.log"
  fi
}

# exp02 runs first because it can finish before the four-view feature pass.
train_and_audit \
  exp02_raw_evidence raw \
  "${CONTENT_FEATURE_DIR}/content_train.parquet" \
  "${CONTENT_FEATURE_DIR}/content_valid.parquet" \
  "${V4_1_EVIDENCE_WEIGHT:-0.20}" 0

MULTIVIEW_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --content-feature-dir "${CONTENT_FEATURE_DIR}"
  --output-dir "${MULTIVIEW_FEATURE_DIR}"
  --batch-size "${V4_1_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V4_1_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V4_1_HASH_BUCKETS:-65536}"
  --tokens-per-view "${V4_1_TOKENS_PER_VIEW:-64}"
  --chars-per-view "${V4_1_CHARS_PER_VIEW:-4096}"
)
[[ "${V4_1_FORCE_MULTIVIEW_PREPARE:-0}" == "1" ]] && MULTIVIEW_PREPARE_ARGS+=(--force)
python -u src/prepare_multiview_content.py "${MULTIVIEW_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_multiview_console.log"

train_and_audit \
  exp01_multiview_standard multiview \
  "${MULTIVIEW_FEATURE_DIR}/multiview_train.parquet" \
  "${MULTIVIEW_FEATURE_DIR}/multiview_valid.parquet" \
  0.0 1

train_and_audit \
  exp03_multiview_evidence multiview \
  "${MULTIVIEW_FEATURE_DIR}/multiview_train.parquet" \
  "${MULTIVIEW_FEATURE_DIR}/multiview_valid.parquet" \
  "${V4_1_EVIDENCE_WEIGHT:-0.20}" 1

COMPARE_ARGS=(--scan-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/comparison.json")
DRAIN_METRICS="${V4_1_DRAIN_METRICS:-artifacts/v1_1_drain/base/metrics.json}"
STRUCTURED_METRICS="${V4_1_STRUCTURED_METRICS:-artifacts/v1_2_structured/base/metrics.json}"
CONTENT_METRICS="${V4_1_CONTENT_METRICS:-artifacts/v3_0_content/exp02_raw_fusion/metrics.json}"
HIERARCHICAL_METRICS="${V4_1_HIERARCHICAL_METRICS:-artifacts/v4_0_hierarchical/exp02_novelty/metrics.json}"
[[ -f "${DRAIN_METRICS}" ]] && COMPARE_ARGS+=(--run "v1.1=${DRAIN_METRICS}")
[[ -f "${STRUCTURED_METRICS}" ]] && COMPARE_ARGS+=(--run "v1.2=${STRUCTURED_METRICS}")
[[ -f "${CONTENT_METRICS}" ]] && COMPARE_ARGS+=(--run "v3.0-exp02=${CONTENT_METRICS}")
[[ -f "${HIERARCHICAL_METRICS}" ]] && COMPARE_ARGS+=(--run "v4.0-exp02=${HIERARCHICAL_METRICS}")
python src/compare_experiments.py "${COMPARE_ARGS[@]}"

echo "v4.1 comparison: ${OUTPUT_ROOT}/comparison.json"
