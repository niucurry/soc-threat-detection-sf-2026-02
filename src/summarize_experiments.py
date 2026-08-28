from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SOC model experiment metrics")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_row(path: Path, root: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    per_class = metrics["per_class"]
    matrix = metrics["confusion_matrix"]["counts"]
    arguments = metrics.get("arguments", {})
    return {
        "run": str(path.parent.relative_to(root)),
        "class_weight_power": arguments.get("class_weight_power"),
        "best_epoch": metrics.get("best_epoch"),
        "competition_score": metrics.get("competition_score"),
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "benign_precision": per_class["benign"]["precision"],
        "benign_recall": per_class["benign"]["recall"],
        "malicious_precision": per_class["malicious"]["precision"],
        "malicious_recall": per_class["malicious"]["recall"],
        "malicious_f1": per_class["malicious"]["f1"],
        "suspicious_precision": per_class["suspicious"]["precision"],
        "suspicious_recall": per_class["suspicious"]["recall"],
        "suspicious_f1": per_class["suspicious"]["f1"],
        "benign_as_malicious": matrix[0][1],
        "malicious_as_benign": matrix[1][0],
        "benign_as_suspicious": matrix[0][2],
        "suspicious_as_benign": matrix[2][0],
    }


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.root.glob("**/metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics.json files found below {args.root}")
    rows = [load_row(path, args.root) for path in metric_paths]
    rows.sort(
        key=lambda row: float(
            row["competition_score"]
            if row["competition_score"] is not None
            else row["macro_f1"]
        ),
        reverse=True,
    )
    output = args.output or args.root / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    print(f"CSV summary: {output}", flush=True)
    print(f"JSON summary: {json_output}", flush=True)


if __name__ == "__main__":
    main()
