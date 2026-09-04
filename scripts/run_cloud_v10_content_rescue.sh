#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# V10 reuses the V9 anchored-residual runner, but reverses the evidence source:
# only a frozen content-branch threat vote may rescue an anchor-benign row.
export V9_OUTPUT_ROOT="${V10_OUTPUT_ROOT:-artifacts/v10_content_rescue}"
export V9_PROCESSED_ROOT="${V10_PROCESSED_ROOT:-data/processed}"
export V9_ANCHOR_MODEL="${V10_ANCHOR_MODEL:-artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/model.pt}"
export V9_V7_PREDICTIONS="${V10_ANCHOR_PREDICTIONS:-artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/valid_predictions.parquet}"
export V9_V7_METRICS="${V10_ANCHOR_METRICS:-artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829/metrics.json}"
export V9_EVIDENCE_SOURCE=content
export V9_EXPERIMENT_VERSION=v10
export V9_ANCHOR_RUN_NAME=cr1_anchor_content
export V9_MULTIVIEW_RUN_NAME=cr2_multiview_content

export V9_EPOCHS="${V10_EPOCHS:-8}"
export V9_BATCH_SIZE="${V10_BATCH_SIZE:-512}"
export V9_SCAN_BATCH_SIZE="${V10_SCAN_BATCH_SIZE:-4096}"
export V9_VALID_BATCH_SIZE="${V10_VALID_BATCH_SIZE:-4096}"
export V9_LEARNING_RATE="${V10_LEARNING_RATE:-0.0005}"
export V9_WEIGHT_DECAY="${V10_WEIGHT_DECAY:-0.0001}"
export V9_DISTILLATION_WEIGHT="${V10_DISTILLATION_WEIGHT:-0.10}"
export V9_TRUST_REGULARIZATION_WEIGHT="${V10_TRUST_REGULARIZATION_WEIGHT:-0.01}"
export V9_CANDIDATE_THREAT_WEIGHT="${V10_CANDIDATE_THREAT_WEIGHT:-1.0}"
export V9_HARD_NEGATIVE_RATIO="${V10_HARD_NEGATIVE_RATIO:-2.0}"
export V9_THREAT_THRESHOLD="${V10_THREAT_THRESHOLD:-0.5}"
export V9_PATIENCE="${V10_PATIENCE:-3}"
export V9_NUM_WORKERS="${V10_NUM_WORKERS:-4}"
export V9_TOKEN_DROPOUT="${V10_TOKEN_DROPOUT:-0.05}"
export V9_RESIDUAL_HIDDEN_DIM="${V10_RESIDUAL_HIDDEN_DIM:-128}"
export V9_MAX_CONFLICT_GAP="${V10_MAX_CONFLICT_GAP:-24}"

exec bash scripts/run_cloud_v9_anchored_residual.sh "${1:-${SOC_DATA_DIR:-data/raw}}"
