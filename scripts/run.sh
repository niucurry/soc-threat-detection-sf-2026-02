#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_DIR}/scripts/run_cloud_v4_0_hierarchical.sh" "$@"
