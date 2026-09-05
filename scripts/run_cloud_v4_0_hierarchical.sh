#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V4_0_OUTPUT_ROOT:-artifacts/v4_0_hierarchical}"
PROCESSED_DIR="${V4_0_PROCESSED_ROOT:-data/processed}"
CONTENT_FEATURE_DIR="${PROCESSED_DIR}/v3_0"
TABULAR_FEATURE_DIR="${PROCESSED_DIR}/v1_0"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CONTENT_FEATURE_DIR}"

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

# v4.0 deliberately reuses v3.0's stateless raw-token features. Re-entering
# the preparer validates completed shards and resumes missing/damaged shards.
PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${TABULAR_FEATURE_DIR}"
  --output-dir "${CONTENT_FEATURE_DIR}"
  --batch-size "${V4_0_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V4_0_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V4_0_HASH_BUCKETS:-65536}"
  --max-tokens "${V4_0_MAX_TOKENS:-96}"
)
if [[ "${V4_0_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
python -u src/prepare_content_features.py "${PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"

GATE_MODES=(none count)
RUN_NAMES=(exp01_no_novelty exp02_novelty)

for index in "${!GATE_MODES[@]}"; do
  gate_mode="${GATE_MODES[$index]}"
  run_name="${RUN_NAMES[$index]}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  if [[ -s "${run_dir}/metrics.json" \
        && -s "${run_dir}/model.pt" \
        && -s "${run_dir}/valid_predictions.parquet" \
        && "${V4_0_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} is complete; reusing it."
  else
    python -u src/train_hierarchical_content.py \
      --experiment-version v4.0 \
      --content-input raw \
      --novelty-gate "${gate_mode}" \
      --train "${CONTENT_FEATURE_DIR}/content_train.parquet" \
      --valid "${CONTENT_FEATURE_DIR}/content_valid.parquet" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V4_0_EPOCHS:-10}" \
      --batch-size "${V4_0_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V4_0_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V4_0_LEARNING_RATE:-0.0007}" \
      --threat-class-weight-power "${V4_0_THREAT_CLASS_WEIGHT_POWER:-0.25}" \
      --subtype-class-weight-power "${V4_0_SUBTYPE_CLASS_WEIGHT_POWER:-0.0}" \
      --subtype-loss-weight "${V4_0_SUBTYPE_LOSS_WEIGHT:-0.75}" \
      --metadata-threat-aux-weight "${V4_0_METADATA_THREAT_AUX_WEIGHT:-0.15}" \
      --content-threat-aux-weight "${V4_0_CONTENT_THREAT_AUX_WEIGHT:-0.25}" \
      --metadata-subtype-aux-weight "${V4_0_METADATA_SUBTYPE_AUX_WEIGHT:-0.35}" \
      --threat-threshold "${V4_0_THREAT_THRESHOLD:-0.5}" \
      --novelty-pseudocount "${V4_0_NOVELTY_PSEUDOCOUNT:-32}" \
      --patience "${V4_0_PATIENCE:-4}" \
      --num-workers "${V4_0_NUM_WORKERS:-4}" \
      --hash-buckets "${V4_0_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V4_0_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V4_0_CONTENT_OUTPUT_DIM:-128}" \
      --token-dropout "${V4_0_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V4_0_CATEGORY_DROPOUT:-0.02}" \
      --seed "${V4_0_SEED:-20260828}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  AUDIT_ARGS=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${CONTENT_FEATURE_DIR}/content_valid.parquet"
    --output-dir "${run_dir}/analysis"
  )
  DRAIN_PREDICTIONS="${V4_0_DRAIN_PREDICTIONS:-artifacts/v1_1_drain/base/valid_predictions.parquet}"
  STRUCTURED_PREDICTIONS="${V4_0_STRUCTURED_PREDICTIONS:-artifacts/v1_2_structured/base/valid_predictions.parquet}"
  CONTENT_PREDICTIONS="${V4_0_CONTENT_PREDICTIONS:-artifacts/v3_0_content/exp02_raw_fusion/valid_predictions.parquet}"
  [[ -f "${DRAIN_PREDICTIONS}" ]] && AUDIT_ARGS+=(--drain-predictions "${DRAIN_PREDICTIONS}")
  [[ -f "${STRUCTURED_PREDICTIONS}" ]] && AUDIT_ARGS+=(--structured-predictions "${STRUCTURED_PREDICTIONS}")
  [[ -f "${CONTENT_PREDICTIONS}" ]] && AUDIT_ARGS+=(--content-predictions "${CONTENT_PREDICTIONS}")
  python src/analyze_hierarchical_errors.py "${AUDIT_ARGS[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
done

COMPARE_ARGS=(--scan-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/comparison.json")
DRAIN_METRICS="${V4_0_DRAIN_METRICS:-artifacts/v1_1_drain/base/metrics.json}"
STRUCTURED_METRICS="${V4_0_STRUCTURED_METRICS:-artifacts/v1_2_structured/base/metrics.json}"
CONTENT_METRICS="${V4_0_CONTENT_METRICS:-artifacts/v3_0_content/exp02_raw_fusion/metrics.json}"
[[ -f "${DRAIN_METRICS}" ]] && COMPARE_ARGS+=(--run "v1.1=${DRAIN_METRICS}")
[[ -f "${STRUCTURED_METRICS}" ]] && COMPARE_ARGS+=(--run "v1.2=${STRUCTURED_METRICS}")
[[ -f "${CONTENT_METRICS}" ]] && COMPARE_ARGS+=(--run "v3.0-exp02=${CONTENT_METRICS}")
python src/compare_experiments.py "${COMPARE_ARGS[@]}"

echo "v4.0 comparison: ${OUTPUT_ROOT}/comparison.json"
echo "Each run contains model.pt, metrics.json, predictions, and an error audit."
