from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.text_specialist import force_rule_probabilities  # noqa: E402


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the text specialist on its routed raw input rows"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.input):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    package = joblib.load(args.model)
    connection = duckdb.connect()
    frame = connection.execute(
        f"""
        SELECT event_id, message_sanitized
        FROM read_parquet('{sql_path(args.input)}')
        WHERE pipeline = 'syslog' AND COALESCE(product_name, '') = ''
        """
    ).fetchdf()
    connection.close()
    text = frame["message_sanitized"].fillna("").astype(str)
    matrix = package["vectorizer"].transform(text)
    classifier = package["classifier"]
    malicious_index = list(classifier.classes_).index("malicious")
    raw_probabilities = classifier.predict_proba(matrix)[:, malicious_index]
    malicious_probabilities, strong_rules = force_rule_probabilities(
        raw_probabilities, text, profile=package["rules_profile"]
    )
    predicted = np.where(
        malicious_probabilities >= float(package["threshold"]),
        "malicious",
        "benign",
    )
    table = pa.table(
        {
            "event_id": frame["event_id"].astype(str).tolist(),
            "pred_label": predicted.tolist(),
            "prob_benign": (1.0 - malicious_probabilities).astype(np.float32),
            "prob_malicious": malicious_probabilities.astype(np.float32),
            "strong_rule": strong_rules,
        }
    )
    pq.write_table(table, args.output, compression="zstd")
    summary = {
        "rows": int(len(frame)),
        "strong_rule_rows": int(strong_rules.sum()),
        "threshold": float(package["threshold"]),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "label_counts": {
            "benign": int(np.sum(predicted == "benign")),
            "malicious": int(np.sum(predicted == "malicious")),
        },
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
