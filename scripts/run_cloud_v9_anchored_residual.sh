#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V9_OUTPUT_ROOT:-artifacts/v9_anchored_residual}"
PROCESSED_DIR="${V9_PROCESSED_ROOT:-data/processed}"
V6_FEATURE_DIR="${PROCESSED_DIR}/v6"
V8_FEATURE_DIR="${PROCESSED_DIR}/v8"
V9_FEATURE_DIR="${PROCESSED_DIR}/v9"
ANCHOR_MODEL="${V9_ANCHOR_MODEL:-artifacts/v7_hierarchical_content/h2_hierarchical_novelty/model.pt}"
EVIDENCE_SOURCE="${V9_EVIDENCE_SOURCE:-metadata}"
EXPERIMENT_VERSION="${V9_EXPERIMENT_VERSION:-v9}"
ANCHOR_RUN_NAME="${V9_ANCHOR_RUN_NAME:-r1_anchor_conflict}"
MULTIVIEW_RUN_NAME="${V9_MULTIVIEW_RUN_NAME:-r2_multiview_conflict}"

if [[ "${EVIDENCE_SOURCE}" != "metadata" && "${EVIDENCE_SOURCE}" != "content" ]]; then
  echo "Unsupported evidence source: ${EVIDENCE_SOURCE}"
  exit 2
fi

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done
if [[ ! -f "${ANCHOR_MODEL}" ]]; then
  echo "Required V7 anchor does not exist: ${ANCHOR_MODEL}"
  echo "Run scripts/run_cloud_v7_hierarchical_content.sh first or set V9_ANCHOR_MODEL."
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${V6_FEATURE_DIR}" "${V8_FEATURE_DIR}" "${V9_FEATURE_DIR}"

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
  --batch-size "${V9_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V9_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V9_HASH_BUCKETS:-65536}"
  --max-tokens "${V9_RAW_MAX_TOKENS:-96}"
)
if [[ "${V9_FORCE_V6_PREPARE:-0}" == "1" ]]; then
  V6_PREPARE_ARGS+=(--force)
fi
python -u src/prepare_v6_content.py "${V6_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_v6_console.log"

V7_PREDICTIONS="${V9_V7_PREDICTIONS:-artifacts/v7_hierarchical_content/h2_hierarchical_novelty/valid_predictions.parquet}"

train_and_audit() {
  local run_name="$1"
  local residual_input="$2"
  local train_path="$3"
  local valid_path="$4"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}" "${run_dir}/analysis"
  if [[ -f "${run_dir}/metrics.json" \
        && -f "${run_dir}/model.pt" \
        && -f "${run_dir}/valid_predictions.parquet" \
        && "${V9_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} model, metrics, and predictions exist; skipping completed experiment."
  else
    python -u src/train_anchored_residual.py \
      --anchor-model "${ANCHOR_MODEL}" \
      --residual-input "${residual_input}" \
      --evidence-source "${EVIDENCE_SOURCE}" \
      --experiment-version "${EXPERIMENT_VERSION}" \
      --train "${train_path}" \
      --valid "${valid_path}" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V9_EPOCHS:-8}" \
      --batch-size "${V9_BATCH_SIZE:-512}" \
      --scan-batch-size "${V9_SCAN_BATCH_SIZE:-4096}" \
      --valid-batch-size "${V9_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V9_LEARNING_RATE:-0.0005}" \
      --weight-decay "${V9_WEIGHT_DECAY:-0.0001}" \
      --distillation-weight "${V9_DISTILLATION_WEIGHT:-0.10}" \
      --trust-regularization-weight "${V9_TRUST_REGULARIZATION_WEIGHT:-0.01}" \
      --candidate-threat-weight "${V9_CANDIDATE_THREAT_WEIGHT:-1.0}" \
      --hard-negative-ratio "${V9_HARD_NEGATIVE_RATIO:-2.0}" \
      --threat-threshold "${V9_THREAT_THRESHOLD:-0.5}" \
      --patience "${V9_PATIENCE:-3}" \
      --num-workers "${V9_NUM_WORKERS:-4}" \
      --content-view-count 4 \
      --content-tokens-per-view "${V9_TOKENS_PER_VIEW:-64}" \
      --token-dropout "${V9_TOKEN_DROPOUT:-0.05}" \
      --residual-hidden-dim "${V9_RESIDUAL_HIDDEN_DIM:-128}" \
      --max-conflict-gap "${V9_MAX_CONFLICT_GAP:-24}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  local audit_args=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${valid_path}"
    --output-dir "${run_dir}/analysis"
  )
  [[ -f "${V7_PREDICTIONS}" ]] && audit_args+=(--v7-predictions "${V7_PREDICTIONS}")
  python src/analyze_v7_errors.py "${audit_args[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
  python src/analyze_v9_residual.py \
    --predictions "${run_dir}/valid_predictions.parquet" \
    --features "${valid_path}" \
    --output-dir "${run_dir}/analysis" \
    --evidence-source "${EVIDENCE_SOURCE}" \
    2>&1 | tee "${run_dir}/residual_console.log"
}

# R1 is the narrow control: it learns whether the configured frozen evidence
# branch is reliable in a final-benign/branch-threat conflict. It needs no V8.
train_and_audit \
  "${ANCHOR_RUN_NAME}" \
  anchor \
  "${V6_FEATURE_DIR}/v6_train.parquet" \
  "${V6_FEATURE_DIR}/v6_valid.parquet"

V8_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --v6-feature-dir "${V6_FEATURE_DIR}"
  --output-dir "${V8_FEATURE_DIR}"
  --batch-size "${V9_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V9_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V9_HASH_BUCKETS:-65536}"
  --tokens-per-view "${V9_TOKENS_PER_VIEW:-64}"
  --chars-per-view "${V9_CHARS_PER_VIEW:-4096}"
)
if [[ "${V9_FORCE_V8_PREPARE:-0}" == "1" ]]; then
  V8_PREPARE_ARGS+=(--force)
fi
python -u src/prepare_v8_content.py "${V8_PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_v8_console.log"

V9_JOIN_ARGS=(
  --v6-feature-dir "${V6_FEATURE_DIR}"
  --v8-feature-dir "${V8_FEATURE_DIR}"
  --output-dir "${V9_FEATURE_DIR}"
)
if [[ "${V9_FORCE_JOIN:-0}" == "1" ]]; then
  V9_JOIN_ARGS+=(--force)
fi
python -u src/prepare_v9_features.py "${V9_JOIN_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_v9_console.log"

# R2 differs from R1 only by adding the transferred head/middle/tail/key-value
# representation to the learned reliability decision.
train_and_audit \
  "${MULTIVIEW_RUN_NAME}" \
  multiview \
  "${V9_FEATURE_DIR}/v9_train.parquet" \
  "${V9_FEATURE_DIR}/v9_valid.parquet"

COMPARE_ARGS=(
  --root "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/comparison.json"
)
V7_METRICS="${V9_V7_METRICS:-artifacts/v7_hierarchical_content/h2_hierarchical_novelty/metrics.json}"
[[ -f "${V7_METRICS}" ]] && COMPARE_ARGS+=(--v7-metrics "${V7_METRICS}")
python src/compare_v7_experiments.py "${COMPARE_ARGS[@]}"

echo "${EXPERIMENT_VERSION} ${EVIDENCE_SOURCE}-evidence comparison: ${OUTPUT_ROOT}/comparison.json"
echo "Each run contains metrics.json, analysis/error_summary.json, and analysis/residual_summary.json."
