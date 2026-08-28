#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

RAW_DATA_DIR="${1:-${SOC_DATA_DIR:-${PROJECT_DIR}/data/raw}}"
for required_file in train.parquet valid_input.parquet valid_answer_private.parquet; do
  if [[ ! -f "${RAW_DATA_DIR}/${required_file}" ]]; then
    echo "Required data file does not exist: ${RAW_DATA_DIR}/${required_file}"
    exit 2
  fi
done

mkdir -p artifacts/v1_npu_tabular data/processed

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee artifacts/v1_npu_tabular/environment.json

if [[ ! -f data/processed/v1_train.parquet || ! -f data/processed/v1_valid.parquet ]]; then
  python src/prepare_features.py \
    --data-dir "${RAW_DATA_DIR}" \
    --output-dir data/processed
else
  echo "Prepared feature files already exist; skipping feature preparation."
fi

python -u src/train_npu_tabular.py \
  --device auto \
  --epochs 20 \
  --batch-size 8192 \
  --learning-rate 0.002 \
  --class-weight-power 0.75 \
  --patience 4 \
  --num-workers 4 \
  --output-dir artifacts/v1_npu_tabular \
  2>&1 | tee artifacts/v1_npu_tabular/train_console.log
