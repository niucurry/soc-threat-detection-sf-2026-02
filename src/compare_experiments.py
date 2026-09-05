from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one comparable table from experiment metrics.json files"
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="NAME=PATH",
        help="Add a named metrics.json file; may be repeated",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        help="Also load immediate child directories containing metrics.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric_value(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def compact(name: str, path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]["counts"]
    errors = sum(sum(row) for row in matrix) - sum(
        matrix[index][index] for index in range(len(matrix))
    )
    audit = metrics.get("hierarchical_audit", {}).get("threat", {})
    per_class = metrics.get("per_class", {})
    return {
        "run": name,
        "metrics_path": str(path),
        "competition_score": metrics["competition_score"],
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "multiclass_log_loss": metrics.get("multiclass_log_loss"),
        "errors": int(errors),
        "benign_recall": per_class.get("benign", {}).get("recall"),
        "malicious_recall": per_class.get("malicious", {}).get("recall"),
        "suspicious_recall": per_class.get("suspicious", {}).get("recall"),
        "threat_false_positive": metric_value(
            audit, "false_positive"
        ),
        "threat_false_negative": metric_value(
            audit, "false_negative"
        ),
        "best_epoch": metrics.get("best_epoch"),
        "model": metrics.get("model"),
        "mode": metrics.get("mode", metrics.get("feature_set")),
    }


def main() -> None:
    args = parse_args()
    named_paths: list[tuple[str, Path]] = list(args.run)
    if args.scan_root is not None:
        named_paths.extend(
            (path.parent.name, path)
            for path in sorted(args.scan_root.glob("*/metrics.json"))
        )

    seen_names: set[str] = set()
    results: list[dict[str, Any]] = []
    for name, path in named_paths:
        if name in seen_names:
            raise ValueError(f"Duplicate run name: {name}")
        seen_names.add(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(compact(name, path))
    if not results:
        raise ValueError("Provide --run and/or --scan-root with at least one result")

    results.sort(
        key=lambda row: (-float(row["competition_score"]), row["errors"], row["run"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
