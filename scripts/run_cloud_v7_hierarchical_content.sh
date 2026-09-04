#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V7_OUTPUT_ROOT:-artifacts/v7_hierarchical_content}"
PROCESSED_DIR="${V7_PROCESSED_DIR:-data/processed}"
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

# V7 deliberately reuses the stateless V6 raw-token shards. Entering the
# preparer on every run validates all completed files and resumes only missing
# or damaged shards after a cloud shutdown.
PREPARE_ARGS=(
  --data-dir "${RAW_DATA_DIR}"
  --base-feature-dir "${PROCESSED_DIR}"
  --output-dir "${V6_FEATURE_DIR}"
  --batch-size "${V7_PREPARE_BATCH_SIZE:-20000}"
  --progress-every "${V7_PROGRESS_EVERY:-100000}"
  --hash-buckets "${V7_HASH_BUCKETS:-65536}"
  --max-tokens "${V7_MAX_TOKENS:-96}"
)
if [[ "${V7_FORCE_PREPARE:-0}" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
python -u src/prepare_v6_content.py "${PREPARE_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/prepare_console.log"

GATE_MODES=(none count)
RUN_NAMES=(h1_hierarchical_raw h2_hierarchical_novelty)

for index in "${!GATE_MODES[@]}"; do
  gate_mode="${GATE_MODES[$index]}"
  run_name="${RUN_NAMES[$index]}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" \
        && -f "${run_dir}/model.pt" \
        && -f "${run_dir}/valid_predictions.parquet" \
        && "${V7_FORCE_TRAIN:-0}" != "1" ]]; then
    echo "${run_name} model, metrics, and predictions exist; skipping completed experiment."
  else
    python -u src/train_hierarchical_content.py \
      --novelty-gate "${gate_mode}" \
      --train "${V6_FEATURE_DIR}/v6_train.parquet" \
      --valid "${V6_FEATURE_DIR}/v6_valid.parquet" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs "${V7_EPOCHS:-10}" \
      --batch-size "${V7_BATCH_SIZE:-2048}" \
      --valid-batch-size "${V7_VALID_BATCH_SIZE:-4096}" \
      --learning-rate "${V7_LEARNING_RATE:-0.0007}" \
      --threat-class-weight-power "${V7_THREAT_CLASS_WEIGHT_POWER:-0.25}" \
      --subtype-class-weight-power "${V7_SUBTYPE_CLASS_WEIGHT_POWER:-0.0}" \
      --subtype-loss-weight "${V7_SUBTYPE_LOSS_WEIGHT:-0.75}" \
      --metadata-threat-aux-weight "${V7_METADATA_THREAT_AUX_WEIGHT:-0.15}" \
      --content-threat-aux-weight "${V7_CONTENT_THREAT_AUX_WEIGHT:-0.25}" \
      --metadata-subtype-aux-weight "${V7_METADATA_SUBTYPE_AUX_WEIGHT:-0.35}" \
      --threat-threshold "${V7_THREAT_THRESHOLD:-0.5}" \
      --novelty-pseudocount "${V7_NOVELTY_PSEUDOCOUNT:-32}" \
      --patience "${V7_PATIENCE:-4}" \
      --num-workers "${V7_NUM_WORKERS:-4}" \
      --hash-buckets "${V7_HASH_BUCKETS:-65536}" \
      --content-embedding-dim "${V7_CONTENT_EMBEDDING_DIM:-64}" \
      --content-output-dim "${V7_CONTENT_OUTPUT_DIM:-128}" \
      --token-dropout "${V7_TOKEN_DROPOUT:-0.05}" \
      --category-dropout "${V7_CATEGORY_DROPOUT:-0.02}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  AUDIT_ARGS=(
    --predictions "${run_dir}/valid_predictions.parquet"
    --features "${V6_FEATURE_DIR}/v6_valid.parquet"
    --output-dir "${run_dir}/analysis"
  )
  V4_PREDICTIONS="${V7_V4_PREDICTIONS:-artifacts/v4_drain_neural/base/valid_predictions.parquet}"
  V5_PREDICTIONS="${V7_V5_PREDICTIONS:-artifacts/v5_structured_neural/base/valid_predictions.parquet}"
  V6_PREDICTIONS="${V7_V6_PREDICTIONS:-artifacts/v6_content_neural/e2_fusion_raw/valid_predictions.parquet}"
  [[ -f "${V4_PREDICTIONS}" ]] && AUDIT_ARGS+=(--v4-predictions "${V4_PREDICTIONS}")
  [[ -f "${V5_PREDICTIONS}" ]] && AUDIT_ARGS+=(--v5-predictions "${V5_PREDICTIONS}")
  [[ -f "${V6_PREDICTIONS}" ]] && AUDIT_ARGS+=(--v6-predictions "${V6_PREDICTIONS}")
  python src/analyze_v7_errors.py "${AUDIT_ARGS[@]}" \
    2>&1 | tee "${run_dir}/analysis_console.log"
done

COMPARE_ARGS=(
  --root "${OUTPUT_ROOT}"
  --output "${OUTPUT_ROOT}/comparison.json"
)
V4_METRICS="${V7_V4_METRICS:-artifacts/v4_drain_neural/base/metrics.json}"
V5_METRICS="${V7_V5_METRICS:-artifacts/v5_structured_neural/base/metrics.json}"
V6_METRICS="${V7_V6_METRICS:-artifacts/v6_content_neural/e2_fusion_raw/metrics.json}"
[[ -f "${V4_METRICS}" ]] && COMPARE_ARGS+=(--v4-metrics "${V4_METRICS}")
[[ -f "${V5_METRICS}" ]] && COMPARE_ARGS+=(--v5-metrics "${V5_METRICS}")
[[ -f "${V6_METRICS}" ]] && COMPARE_ARGS+=(--v6-metrics "${V6_METRICS}")
python src/compare_v7_experiments.py "${COMPARE_ARGS[@]}"

echo "V7 comparison: ${OUTPUT_ROOT}/comparison.json"
echo "Each run contains metrics.json, valid_predictions.parquet, and analysis/error_summary.json."
