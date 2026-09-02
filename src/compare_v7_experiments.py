from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V7 hierarchical experiments")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v4-metrics", type=Path)
    parser.add_argument("--v5-metrics", type=Path)
    parser.add_argument("--v6-metrics", type=Path)
    return parser.parse_args()


def compact(name: str, path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    matrix = metrics["confusion_matrix"]["counts"]
    audit = metrics.get("hierarchical_audit", {})
    threat = audit.get("threat", metrics.get("competition_metrics", {}))
    return {
        "run": name,
        "metrics_path": str(path),
        "competition_score": metrics["competition_score"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "multiclass_log_loss": metrics.get("multiclass_log_loss"),
        "errors": int(
            sum(sum(row) for row in matrix) - sum(matrix[i][i] for i in range(3))
        ),
        "benign_recall": metrics["per_class"]["benign"]["recall"],
        "malicious_recall": metrics["per_class"]["malicious"]["recall"],
        "suspicious_recall": metrics["per_class"]["suspicious"]["recall"],
        "threat_f1": threat.get("f1", threat.get("threat_binary_f1")),
        "threat_recall": threat.get("recall", threat.get("threat_binary_recall")),
        "threat_false_positive": threat.get("false_positive"),
        "threat_false_negative": threat.get("false_negative"),
        "subtype_accuracy": audit.get("subtype_accuracy_on_true_threat"),
        "unseen_combo_errors": audit.get("unseen_combo_errors"),
        "best_epoch": metrics.get("best_epoch"),
        "mode": metrics.get("mode", metrics.get("feature_set", name)),
    }


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    baselines = (
        ("v4", args.v4_metrics),
        ("v5", args.v5_metrics),
        ("v6_e2", args.v6_metrics),
    )
    for name, path in baselines:
        if path is not None and path.is_file():
            results.append(compact(name, path))
    for path in sorted(args.root.glob("*/metrics.json")):
        results.append(compact(path.parent.name, path))
    if not results:
        raise FileNotFoundError(f"No metrics.json files found below {args.root}")
    results.sort(key=lambda value: float(value["competition_score"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
