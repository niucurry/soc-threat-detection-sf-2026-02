#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_PROJECT_DIR="$(cd "${1:-${PROJECT_DIR}}" && pwd)"
cd "${PROJECT_DIR}"

link_file() {
  local source_path="$1"
  local target_path="$2"
  if [[ ! -e "${source_path}" ]]; then
    echo "skip missing legacy file: ${source_path}"
    return
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -e "${target_path}" || -L "${target_path}" ]]; then
    echo "keep existing canonical path: ${target_path}"
    return
  fi
  ln -s "${source_path}" "${target_path}"
  echo "linked: ${target_path} -> ${source_path}"
}

link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v1_train.parquet" \
  "${PROJECT_DIR}/data/processed/v1_0/tabular_train.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v1_valid.parquet" \
  "${PROJECT_DIR}/data/processed/v1_0/tabular_valid.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v1_manifest.json" \
  "${PROJECT_DIR}/data/processed/v1_0/tabular_manifest.json"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v6/v6_train.parquet" \
  "${PROJECT_DIR}/data/processed/v3_0/content_train.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v6/v6_valid.parquet" \
  "${PROJECT_DIR}/data/processed/v3_0/content_valid.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v6/v6_manifest.json" \
  "${PROJECT_DIR}/data/processed/v3_0/content_manifest.json"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v8/v8_train.parquet" \
  "${PROJECT_DIR}/data/processed/v4_1/multiview_train.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v8/v8_valid.parquet" \
  "${PROJECT_DIR}/data/processed/v4_1/multiview_valid.parquet"
link_file \
  "${LEGACY_PROJECT_DIR}/data/processed/v8/v8_manifest.json" \
  "${PROJECT_DIR}/data/processed/v4_1/multiview_manifest.json"

LEGACY_ANCHOR="${LEGACY_PROJECT_DIR}/artifacts/v7_hierarchical_content/h2_recovery_oldv6_seed20260829"
CANONICAL_ANCHOR="${PROJECT_DIR}/artifacts/v4_0_hierarchical/exp02_novelty-seed20260829"
for artifact in model.pt metrics.json valid_predictions.parquet preprocessor.json; do
  link_file "${LEGACY_ANCHOR}/${artifact}" "${CANONICAL_ANCHOR}/${artifact}"
done

echo "Legacy links are read-only aliases; source files were not copied or deleted."
