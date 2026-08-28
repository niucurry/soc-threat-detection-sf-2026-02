#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
OUTPUT_ROOT="${V2_OUTPUT_ROOT:-artifacts/v2_hybrid}"
for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p \
  "${OUTPUT_ROOT}/base" \
  "${OUTPUT_ROOT}/text" \
  "${OUTPUT_ROOT}/validation_conservative" \
  "${OUTPUT_ROOT}/validation_tuned" \
  data/processed

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee "${OUTPUT_ROOT}/environment.json"

if [[ ! -f data/processed/v1_train.parquet || ! -f data/processed/v1_valid.parquet ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir data/processed
else
  echo "Prepared feature files already exist; skipping feature preparation."
fi

python -u src/train_npu_tabular.py \
  --train data/processed/v1_train.parquet \
  --valid data/processed/v1_valid.parquet \
  --device auto \
  --epochs 12 \
  --batch-size 8192 \
  --learning-rate 0.002 \
  --class-weight-power 0.0 \
  --patience 3 \
  --num-workers 4 \
  --output-dir "${OUTPUT_ROOT}/base" \
  2>&1 | tee "${OUTPUT_ROOT}/base/train_console.log"

python -u src/train_text_specialist.py \
  --train-raw "${RAW_DATA_DIR}/train.parquet" \
  --valid-input "${RAW_DATA_DIR}/valid_input.parquet" \
  --valid-answer "${RAW_DATA_DIR}/valid_answer_private.parquet" \
  --base-predictions "${OUTPUT_ROOT}/base/valid_predictions.parquet" \
  --output-dir "${OUTPUT_ROOT}/text" \
  2>&1 | tee "${OUTPUT_ROOT}/text/train_console.log"

python src/combine_v2_predictions.py \
  --base-predictions "${OUTPUT_ROOT}/base/valid_predictions.parquet" \
  --specialist-predictions "${OUTPUT_ROOT}/text/valid_predictions.parquet" \
  --output-dir "${OUTPUT_ROOT}/validation_conservative"

python src/combine_v2_predictions.py \
  --base-predictions "${OUTPUT_ROOT}/base/valid_predictions.parquet" \
  --specialist-predictions "${OUTPUT_ROOT}/text/valid_predictions.parquet" \
  --raw-input "${RAW_DATA_DIR}/valid_input.parquet" \
  --output-dir "${OUTPUT_ROOT}/validation_tuned"
