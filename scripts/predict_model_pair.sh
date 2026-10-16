#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: bash scripts/predict_model_pair.sh TEST_INPUT.csv OLD_BACKUP.tar.gz NEW_MODEL.pt OUTPUT_DIR"
  exit 2
fi
# Resolve arguments before changing directories, so relative input paths are unambiguous.
TEST_CSV="$(realpath -e -- "$1")"
OLD_BACKUP="$(realpath -e -- "$2")"
NEW_MODEL="$(realpath -e -- "$3")"
PAIR_DIR="$(realpath -m -- "$4")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for required in "$TEST_CSV" "$OLD_BACKUP" "$OLD_BACKUP.sha256" "$NEW_MODEL" "$NEW_MODEL.sha256"; do
  [[ -f "$required" ]] || { echo "Required file missing: $required"; exit 2; }
done
mkdir -p "$PAIR_DIR"
exec 8>"$PAIR_DIR/pair.lock"
flock -n 8 || { echo "This comparison is already running"; exit 2; }
cd "$PROJECT_DIR"
python -u src/prepare_prediction_csv.py --input "$TEST_CSV" --output-dir "$PAIR_DIR/input"
# Sequential execution avoids two models competing for the same NPU's memory.
# Both load their own saved preprocessing parameters; neither fits anything to the test set.
bash scripts/predict_test.sh "$PAIR_DIR/input/test_input.parquet" "$OLD_BACKUP" "$PAIR_DIR/old_model"
bash scripts/predict_full_label_refit.sh "$PAIR_DIR/input/test_input.parquet" "$NEW_MODEL" "$PAIR_DIR/new_model"
python -u src/compare_prediction_files.py --input "$PAIR_DIR/input/test_input.parquet" \
  --old-dir "$PAIR_DIR/old_model" --new-dir "$PAIR_DIR/new_model" --output-dir "$PAIR_DIR/comparison"
echo "Old pure-model submission: $PAIR_DIR/old_model/res.csv"
echo "New pure-model submission: $PAIR_DIR/new_model/res.csv"
echo "Comparison summary: $PAIR_DIR/comparison/comparison.json"
echo "No manual label overrides or platform submission were performed."
