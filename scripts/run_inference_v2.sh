#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

TEST_INPUT="${1:?Usage: bash scripts/run_inference_v2.sh /path/to/test.parquet [res.csv]}"
RESULT_PATH="${2:-artifacts/v2_submission/res.csv}"
MODEL_ROOT="${V2_MODEL_ROOT:-artifacts/v2_hybrid}"
INFERENCE_DIR="${V2_INFERENCE_DIR:-artifacts/v2_submission}"
RULE_MODE="${V2_RULE_MODE:-tuned}"

for required_file in \
  "${TEST_INPUT}" \
  "${MODEL_ROOT}/base/model.pt" \
  "${MODEL_ROOT}/text/model.joblib"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}"
    exit 2
  fi
done

mkdir -p "${INFERENCE_DIR}" "$(dirname "${RESULT_PATH}")" data/processed

python src/prepare_inference_features.py \
  --input "${TEST_INPUT}" \
  --output data/processed/v2_test.parquet \
  --force

python src/predict_tabular_checkpoint.py \
  --model "${MODEL_ROOT}/base/model.pt" \
  --data data/processed/v2_test.parquet \
  --output "${INFERENCE_DIR}/base_predictions.parquet" \
  --device auto \
  --num-workers 4 \
  --force

python src/predict_text_specialist.py \
  --model "${MODEL_ROOT}/text/model.joblib" \
  --input "${TEST_INPUT}" \
  --output "${INFERENCE_DIR}/specialist_predictions.parquet" \
  --force

RULE_ARGS=()
if [[ "${RULE_MODE}" == "conservative" ]]; then
  RULE_ARGS+=(--disable-suspicious-rules)
elif [[ "${RULE_MODE}" != "tuned" ]]; then
  echo "V2_RULE_MODE must be tuned or conservative, got: ${RULE_MODE}"
  exit 2
fi

python src/make_v2_submission.py \
  --base-predictions "${INFERENCE_DIR}/base_predictions.parquet" \
  --specialist-predictions "${INFERENCE_DIR}/specialist_predictions.parquet" \
  --raw-input "${TEST_INPUT}" \
  --output "${RESULT_PATH}" \
  --force \
  "${RULE_ARGS[@]}"
