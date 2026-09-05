#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
MODEL_VERSION="${V5_MODEL_VERSION:-v5.0}"
OUTPUT_ROOT="${V5_OUTPUT_ROOT:-artifacts/v5_0_metadata_residual}"
PROCESSED_DIR="${V5_PROCESSED_ROOT:-data/processed}"
CONTENT_FEATURE_DIR="${V5_CONTENT_FEATURE_DIR:-${PROCESSED_DIR}/v3_0}"
MULTIVIEW_FEATURE_DIR="${V5_MULTIVIEW_FEATURE_DIR:-${PROCESSED_DIR}/v4_1}"
RESIDUAL_FEATURE_DIR="${V5_RESIDUAL_FEATURE_DIR:-${PROCESSED_DIR}/v5}"
TABULAR_FEATURE_DIR="${PROCESSED_DIR}/v1_0"
ANCHOR_MODEL="${V5_ANCHOR_MODEL:-artifacts/v4_0_hierarchical/exp02_novelty-seed20260829/model.pt}"
ANCHOR_PREDICTIONS="${V5_ANCHOR_PREDICTIONS:-artifacts/v4_0_hierarchical/exp02_novelty-seed20260829/valid_predictions.parquet}"
ANCHOR_METRICS="${V5_ANCHOR_METRICS:-artifacts/v4_0_hierarchical/exp02_novelty-seed20260829/metrics.json}"
EVIDENCE_SOURCE="${V5_EVIDENCE_SOURCE:-metadata}"
MAX_CONFLICT_GAP="${V5_MAX_CONFLICT_GAP:-24}"
ANCHOR_RUN_NAME="${V5_ANCHOR_RUN_NAME:-exp01_anchor_metadata}"
MULTIVIEW_RUN_NAME="${V5_MULTIVIEW_RUN_NAME:-exp02_multiview_metadata}"
RUN_MULTIVIEW="${V5_RUN_MULTIVIEW:-0}"

case "${MODEL_VERSION}" in
  v5.0) ;;
  *) echo "Unsupported model version: ${MODEL_VERSION}"; exit 2 ;;
esac
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
  echo "Required frozen v4.0 anchor does not exist: ${ANCHOR_MODEL}"
  echo "Run scripts/run_cloud_v4_0_seed_sweep.sh first, or set V5_ANCHOR_MODEL."
  exit 2
fi

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${CONTENT_FEATURE_DIR}" \
  "${MULTIVIEW_FEATURE_DIR}" \
  "${RESIDUAL_FEATURE_DIR}"
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
  --batch-size "${V5_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V5_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V5_HASH_BUCKETS:-65536}"
  --max-tokens "${V5_RAW_MAX_TOKENS:-96}"
)
if [[ "${V5_REUSE_CONTENT_FILES:-0}" == "1" ]]; then
  for feature_file in content_train.parquet content_valid.parquet; do
    if [[ ! -f "${CONTENT_FEATURE_DIR}/${feature_file}" ]]; then
      echo "V5_REUSE_CONTENT_FILES=1 but file is missing: ${CONTENT_FEATURE_DIR}/${feature_file}"
      exit 2
    fi
  done
  echo "Reusing explicitly supplied v3.0 content files without rebuilding shards."
else
  [[ "${V5_FORCE_CONTENT_PREPARE:-0}" == "1" ]] && CONTENT_PREPARE_ARGS+=(--force)
  python -u src/prepare_content_features.py "${CONTENT_PREPARE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/prepare_content_console.log"
fi

train_and_audit() {
  local run_name="$1"
  local residual_input="$2"
  local train_path="$3"
  local valid_path="$4"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}/analysis"

  if [[ -s "${run_dir}/metrics.json" \
        && -s "${run_dir}/model.pt" \
        && -s "${run_dir}/valid_predictions.parquet" \
        && "${V5_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} is complete; reusing it."
  else
    python -u src/train_anchored_residual.py \
      --anchor-model "${ANCHOR_MODEL}" \
      --residual-input "${residual_input}" \
      --evidence-source "${EVIDENCE_SOURCE}" \
      --experiment-version "${MODEL_VERSION}" \
      --train "${train_path}" \
      --valid "${valid_path}" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V5_EPOCHS:-8}" \
      --batch-size "${V5_BATCH_SIZE:-512}" \
      --scan-batch-size "${V5_SCAN_BATCH_SIZE:-4096}" \
      --valid-batch-size "${V5_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V5_LEARNING_RATE:-0.0005}" \
      --weight-decay "${V5_WEIGHT_DECAY:-0.0001}" \
      --distillation-weight "${V5_DISTILLATION_WEIGHT:-0.10}" \
      --trust-regularization-weight "${V5_TRUST_REGULARIZATION_WEIGHT:-0.01}" \
      --candidate-threat-weight "${V5_CANDIDATE_THREAT_WEIGHT:-1.0}" \
      --hard-negative-ratio "${V5_HARD_NEGATIVE_RATIO:-2.0}" \
      --threat-threshold "${V5_THREAT_THRESHOLD:-0.5}" \
      --patience "${V5_PATIENCE:-3}" \
      --num-workers "${V5_NUM_WORKERS:-4}" \
      --content-view-count 4 \
      --content-tokens-per-view "${V5_TOKENS_PER_VIEW:-64}" \
      --token-dropout "${V5_TOKEN_DROPOUT:-0.05}" \
      --residual-hidden-dim "${V5_RESIDUAL_HIDDEN_DIM:-128}" \
      --max-conflict-gap "${MAX_CONFLICT_GAP}" \
      --seed "${V5_SEED:-20260904}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  AUDIT_ARGS=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${valid_path}"
    --output-dir "${run_dir}/analysis"
  )
  [[ -f "${ANCHOR_PREDICTIONS}" ]] && AUDIT_ARGS+=(--hierarchical-predictions "${ANCHOR_PREDICTIONS}")
  python src/analyze_hierarchical_errors.py "${AUDIT_ARGS[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
  python src/analyze_anchored_residual.py \
    --predictions "${run_dir}/valid_predictions.parquet" \
    --features "${valid_path}" \
    --output-dir "${run_dir}/analysis" \
    --evidence-source "${EVIDENCE_SOURCE}" \
    2>&1 | tee "${run_dir}/residual_console.log"
}

# exp01 learns whether the selected frozen evidence source is trustworthy
# when it conflicts with an anchor-benign prediction. See this branch's README.
train_and_audit \
  "${ANCHOR_RUN_NAME}" anchor \
  "${CONTENT_FEATURE_DIR}/content_train.parquet" \
  "${CONTENT_FEATURE_DIR}/content_valid.parquet"

if [[ "${RUN_MULTIVIEW}" == "1" ]]; then
MULTIVIEW_PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --content-feature-dir "${CONTENT_FEATURE_DIR}"
  --output-dir "${MULTIVIEW_FEATURE_DIR}"
  --batch-size "${V5_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V5_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V5_HASH_BUCKETS:-65536}"
  --tokens-per-view "${V5_TOKENS_PER_VIEW:-64}"
  --chars-per-view "${V5_CHARS_PER_VIEW:-4096}"
)
[[ "${V5_FORCE_MULTIVIEW_PREPARE:-0}" == "1" ]] && MULTIVIEW_PREPARE_ARGS+=(--force)
if [[ "${V5_REUSE_MULTIVIEW_FILES:-0}" == "1" ]]; then
  for feature_file in multiview_train.parquet multiview_valid.parquet; do
    if [[ ! -f "${MULTIVIEW_FEATURE_DIR}/${feature_file}" ]]; then
      echo "V5_REUSE_MULTIVIEW_FILES=1 but file is missing: ${MULTIVIEW_FEATURE_DIR}/${feature_file}"
      exit 2
    fi
  done
else
  python -u src/prepare_multiview_content.py "${MULTIVIEW_PREPARE_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/prepare_multiview_console.log"
fi

RESIDUAL_JOIN_ARGS=(
  --content-feature-dir "${CONTENT_FEATURE_DIR}"
  --multiview-feature-dir "${MULTIVIEW_FEATURE_DIR}"
  --output-dir "${RESIDUAL_FEATURE_DIR}"
)
[[ "${V5_FORCE_JOIN:-0}" == "1" ]] && RESIDUAL_JOIN_ARGS+=(--force)
python -u src/prepare_residual_features.py "${RESIDUAL_JOIN_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_residual_console.log"

# exp02 tests whether four trainable content views improve the trust decision.
# This optional ablation is separate from the default exp01 experiment;
# its measured results and limitations are recorded in this branch's README.
train_and_audit \
  "${MULTIVIEW_RUN_NAME}" multiview \
  "${RESIDUAL_FEATURE_DIR}/residual_train.parquet" \
  "${RESIDUAL_FEATURE_DIR}/residual_valid.parquet"
fi

COMPARE_ARGS=(--scan-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/comparison.json")
[[ -f "${ANCHOR_METRICS}" ]] && COMPARE_ARGS+=(--run "v4.0-exp02-seed20260829=${ANCHOR_METRICS}")
python src/compare_experiments.py "${COMPARE_ARGS[@]}"

echo "${MODEL_VERSION} ${EVIDENCE_SOURCE}-evidence comparison: ${OUTPUT_ROOT}/comparison.json"
echo "Official default run: ${OUTPUT_ROOT}/${ANCHOR_RUN_NAME}"
[[ "${RUN_MULTIVIEW}" == "0" ]] && echo "Set V5_RUN_MULTIVIEW=1 to run exp02."
