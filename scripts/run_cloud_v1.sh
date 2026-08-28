#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

mkdir -p artifacts/v1_npu_tabular data/processed

python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-npu.txt

python src/check_cloud_env.py | tee artifacts/v1_npu_tabular/environment.json

if [[ ! -f data/processed/v1_train.parquet || ! -f data/processed/v1_valid.parquet ]]; then
  python src/prepare_features.py \
    --data-dir data/raw \
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
