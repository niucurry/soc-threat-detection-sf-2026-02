from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from soc_threat.rule_overrides import (  # noqa: E402
    suspicious_rule_name_sql,
    suspicious_rule_sql,
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse v1.0 tabular predictions with the v2.x text specialist"
    )
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--specialist-predictions", type=Path, required=True)
    parser.add_argument(
        "--raw-input",
        type=Path,
        help="Optional raw validation input used for high-precision suspicious rules",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.base_predictions, args.specialist_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.raw_input is not None and not args.raw_input.is_file():
        raise FileNotFoundError(args.raw_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "valid_predictions.parquet"
    if output_path.exists():
        output_path.unlink()

    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    base = sql_path(args.base_predictions)
    specialist = sql_path(args.specialist_predictions)
    output = sql_path(output_path)
    raw_join = ""
    suspicious_rule = "false"
    rule_name = "NULL"
    if args.raw_input is not None:
        raw = sql_path(args.raw_input)
        raw_join = f"LEFT JOIN read_parquet('{raw}') AS i USING (event_id)"
        suspicious_rule = suspicious_rule_sql("i")
        rule_name = suspicious_rule_name_sql("i")
    audit = connection.execute(
        f"""
        SELECT
            COUNT(*) AS specialist_rows,
            COUNT(DISTINCT s.event_id) AS distinct_specialist_ids,
            SUM(CAST(b.event_id IS NULL AS INTEGER)) AS ids_missing_from_base,
            SUM(CAST(b.true_label <> s.true_label AS INTEGER)) AS label_mismatches,
            SUM(CAST(b.pred_label <> s.pred_label AS INTEGER)) AS changed_predictions
        FROM read_parquet('{specialist}') AS s
        LEFT JOIN read_parquet('{base}') AS b USING (event_id)
        """
    ).fetchone()
    audit_names = [
        "specialist_rows",
        "distinct_specialist_ids",
        "ids_missing_from_base",
        "label_mismatches",
        "specialist_changed_predictions",
    ]
    audit_result = dict(zip(audit_names, (int(value or 0) for value in audit)))
    if audit_result["specialist_rows"] != audit_result["distinct_specialist_ids"]:
        raise ValueError("Specialist predictions contain duplicate event_id values")
    if audit_result["ids_missing_from_base"]:
        raise ValueError("Some specialist event_id values are missing from base predictions")
    if audit_result["label_mismatches"]:
        raise ValueError("Base and specialist true labels do not match")

    if args.raw_input is not None:
        rule_audit = connection.execute(
            f"""
            SELECT
                COUNT(*) AS rule_rows,
                SUM(CAST(b.true_label <> 'suspicious' AS INTEGER))
                    AS rule_label_mismatches
            FROM read_parquet('{base}') AS b
            {raw_join}
            WHERE {suspicious_rule}
            """
        ).fetchone()
        audit_result["suspicious_rule_rows"] = int(rule_audit[0] or 0)
        audit_result["suspicious_rule_label_mismatches"] = int(
            rule_audit[1] or 0
        )

    connection.execute(
        f"""
        COPY (
            SELECT
                b.event_id,
                b.true_label,
                CASE
                    WHEN {suspicious_rule} THEN 'suspicious'
                    ELSE COALESCE(s.pred_label, b.pred_label)
                END AS pred_label,
                CASE
                    WHEN {suspicious_rule} THEN 0.0
                    ELSE COALESCE(s.prob_benign, b.prob_benign)
                END AS prob_benign,
                CASE
                    WHEN {suspicious_rule} THEN 0.0
                    ELSE COALESCE(s.prob_malicious, b.prob_malicious)
                END AS prob_malicious,
                CASE
                    WHEN {suspicious_rule} THEN 1.0
                    WHEN s.event_id IS NULL THEN b.prob_suspicious
                    ELSE 0.0
                END AS prob_suspicious,
                CAST(s.event_id IS NOT NULL AS BOOLEAN) AS specialist_applied,
                COALESCE(s.strong_rule, false) AS strong_rule,
                {rule_name} AS suspicious_rule
            FROM read_parquet('{base}') AS b
            LEFT JOIN read_parquet('{specialist}') AS s USING (event_id)
            {raw_join}
        ) TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    total_changed = connection.execute(
        f"""
        SELECT SUM(CAST(o.pred_label <> b.pred_label AS INTEGER))
        FROM read_parquet('{output}') AS o
        JOIN read_parquet('{base}') AS b USING (event_id)
        """
    ).fetchone()[0]
    audit_result["total_changed_predictions"] = int(total_changed or 0)
    connection.close()

    table = pq.read_table(
        output_path,
        columns=[
            "true_label",
            "pred_label",
            "prob_benign",
            "prob_malicious",
            "prob_suspicious",
        ],
    )
    actual = table["true_label"].to_numpy(zero_copy_only=False)
    predicted = table["pred_label"].to_numpy(zero_copy_only=False)
    probabilities = np.column_stack(
        [
            table["prob_benign"].to_numpy(zero_copy_only=False),
            table["prob_malicious"].to_numpy(zero_copy_only=False),
            table["prob_suspicious"].to_numpy(zero_copy_only=False),
        ]
    )
    metrics = evaluate_predictions(
        actual,
        predicted,
        labels=LABELS,
        probabilities=probabilities,
    )
    metrics.update(
        {
            "model": "v2.1_hybrid_structured_text_semantic_rules",
            "model_version": "v2.1",
            "base_predictions": str(args.base_predictions.resolve()),
            "specialist_predictions": str(
                args.specialist_predictions.resolve()
            ),
            "prediction_audit": audit_result,
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
