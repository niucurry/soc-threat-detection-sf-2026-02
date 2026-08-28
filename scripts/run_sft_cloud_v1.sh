#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

SFT_CSV="${1:-${SFT_CSV_PATH:-}}"
if [[ -z "${SFT_CSV}" ]]; then
  echo "Usage: bash scripts/run_sft_cloud_v1.sh /absolute/path/train_system_prompt_response.csv"
  exit 2
fi
if [[ ! -f "${SFT_CSV}" ]]; then
  echo "SFT CSV does not exist: ${SFT_CSV}"
  exit 2
fi

mkdir -p artifacts/v1_sft_npu_tabular data/processed

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py \
  | tee artifacts/v1_sft_npu_tabular/environment.json

if [[ ! -f data/processed/sft_v1_train.parquet || ! -f data/processed/sft_v1_valid.parquet ]]; then
  python -u src/prepare_sft_features.py \
    --input "${SFT_CSV}" \
    --output-dir data/processed \
    --valid-percent 10
else
  echo "Prepared SFT feature files already exist; skipping feature preparation."
fi

python -u src/train_npu_tabular.py \
  --train data/processed/sft_v1_train.parquet \
  --valid data/processed/sft_v1_valid.parquet \
  --device auto \
  --epochs 15 \
  --batch-size 8192 \
  --learning-rate 0.002 \
  --class-weight-power 0.5 \
  --patience 3 \
  --num-workers 4 \
  --output-dir artifacts/v1_sft_npu_tabular \
  2>&1 | tee artifacts/v1_sft_npu_tabular/train_console.log

