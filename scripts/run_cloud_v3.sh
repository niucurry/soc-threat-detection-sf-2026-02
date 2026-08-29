#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
V2_ROOT="${V2_OUTPUT_ROOT:-artifacts/v2_hybrid}"
OUTPUT_ROOT="${V3_OUTPUT_ROOT:-artifacts/v3_semantic_rules}"
BASE_PREDICTIONS="${V3_BASE_PREDICTIONS:-${V2_ROOT}/base/valid_predictions.parquet}"

for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done
if [[ ! -f "${BASE_PREDICTIONS}" ]]; then
  echo "V2 base predictions do not exist: ${BASE_PREDICTIONS}"
  echo "Run scripts/run_cloud_v2.sh first, or set V3_BASE_PREDICTIONS."
  exit 2
fi

mkdir -p \
  "${OUTPUT_ROOT}/text" \
  "${OUTPUT_ROOT}/validation_conservative" \
  "${OUTPUT_ROOT}/validation_tuned"

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

python -u src/train_text_specialist.py \
  --train-raw "${RAW_DATA_DIR}/train.parquet" \
  --valid-input "${RAW_DATA_DIR}/valid_input.parquet" \
  --valid-answer "${RAW_DATA_DIR}/valid_answer_private.parquet" \
  --base-predictions "${BASE_PREDICTIONS}" \
  --output-dir "${OUTPUT_ROOT}/text" \
  2>&1 | tee "${OUTPUT_ROOT}/text/train_console.log"

python src/combine_v2_predictions.py \
  --base-predictions "${BASE_PREDICTIONS}" \
  --specialist-predictions "${OUTPUT_ROOT}/text/valid_predictions.parquet" \
  --output-dir "${OUTPUT_ROOT}/validation_conservative"

python src/combine_v2_predictions.py \
  --base-predictions "${BASE_PREDICTIONS}" \
  --specialist-predictions "${OUTPUT_ROOT}/text/valid_predictions.parquet" \
  --raw-input "${RAW_DATA_DIR}/valid_input.parquet" \
  --output-dir "${OUTPUT_ROOT}/validation_tuned"

echo "V3 semantic-rule validation completed: ${OUTPUT_ROOT}"
