#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

FEATURE_DIR="${1:-${V4_0_SWEEP_FEATURE_DIR:-data/processed/v3_0}}"
OUTPUT_ROOT="${2:-${V4_0_SWEEP_OUTPUT_ROOT:-artifacts/v4_0_hierarchical}}"
SEEDS="${V4_0_SWEEP_SEEDS:-20260828 20260829 20260830 20260831}"
TARGET_ERRORS="${V4_0_SWEEP_TARGET_ERRORS:-46}"
TARGET_SCORE="${V4_0_SWEEP_TARGET_SCORE:-0.9993140394289346}"

TRAIN_FILE="${FEATURE_DIR}/content_train.parquet"
VALID_FILE="${FEATURE_DIR}/content_valid.parquet"

for required_file in "${TRAIN_FILE}" "${VALID_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required v3.0 content feature file does not exist: ${required_file}"
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  run_dir="${OUTPUT_ROOT}/exp02_novelty-seed${seed}"
  mkdir -p "${run_dir}"

  if [[ -s "${run_dir}/model.pt" \
        && -s "${run_dir}/metrics.json" \
        && -s "${run_dir}/valid_predictions.parquet" ]]; then
    echo "seed=${seed}: complete artifacts already exist; reusing them."
  else
    echo "seed=${seed}: starting v4.0-exp02 in ${run_dir}"
    python -u src/train_hierarchical_content.py \
      --experiment-version v4.0 \
      --content-input raw \
      --novelty-gate count \
      --train "${TRAIN_FILE}" \
      --valid "${VALID_FILE}" \
      --output-dir "${run_dir}" \
      --device auto \
      --epochs 10 \
      --batch-size 2048 \
      --valid-batch-size 4096 \
      --learning-rate 0.0007 \
      --threat-class-weight-power 0.25 \
      --subtype-class-weight-power 0.0 \
      --subtype-loss-weight 0.75 \
      --metadata-threat-aux-weight 0.15 \
      --content-threat-aux-weight 0.25 \
      --metadata-subtype-aux-weight 0.35 \
      --threat-threshold 0.5 \
      --novelty-pseudocount 32 \
      --patience 4 \
      --num-workers 4 \
      --hash-buckets 65536 \
      --content-embedding-dim 64 \
      --content-output-dim 128 \
      --token-dropout 0.05 \
      --category-dropout 0.02 \
      --seed "${seed}" \
      2>&1 | tee "${run_dir}/train_console.log"
  fi

  if python - "${run_dir}/metrics.json" "${TARGET_ERRORS}" "${TARGET_SCORE}" <<'PY'
import json
import sys

metrics_path, target_errors, target_score = sys.argv[1:]
with open(metrics_path, encoding="utf-8") as handle:
    metrics = json.load(handle)
counts = metrics["confusion_matrix"]["counts"]
errors = sum(map(sum, counts)) - sum(counts[i][i] for i in range(len(counts)))
score = float(metrics["competition_score"])
print(
    f"result={metrics_path} score={score:.16f} errors={errors} "
    f"best_epoch={metrics['best_epoch']}"
)
raise SystemExit(
    0 if errors <= int(target_errors) and score >= float(target_score) else 1
)
PY
  then
    echo "Target reached at seed=${seed}; stopping the sweep early."
    break
  fi
done

COMPARE_ARGS=(--output "${OUTPUT_ROOT}/exp02_seed_comparison.json")
for seed in ${SEEDS}; do
  metrics_path="${OUTPUT_ROOT}/exp02_novelty-seed${seed}/metrics.json"
  [[ -f "${metrics_path}" ]] && COMPARE_ARGS+=(--run "v4.0-exp02-seed${seed}=${metrics_path}")
done
python src/compare_experiments.py "${COMPARE_ARGS[@]}"
