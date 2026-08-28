from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.rule_overrides import suspicious_rule_sql  # noqa: E402


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine V2 test predictions and write a validated res.csv"
    )
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--specialist-predictions", type=Path, required=True)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disable-suspicious-rules",
        action="store_true",
        help="Use the conservative model-only result for non-specialist rows",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.base_predictions,
        args.specialist_predictions,
        args.raw_input,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        if not args.force:
            raise FileExistsError(
                f"{args.output} already exists; pass --force to replace it"
            )
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base = sql_path(args.base_predictions)
    specialist = sql_path(args.specialist_predictions)
    raw = sql_path(args.raw_input)
    output = sql_path(args.output)
    rule = "false" if args.disable_suspicious_rules else suspicious_rule_sql("i")
    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute(
        f"""
        CREATE VIEW final_predictions AS
        SELECT
            b.event_id,
            CASE
                WHEN {rule} THEN 'suspicious'
                ELSE COALESCE(s.pred_label, b.pred_label)
            END AS pred_label
        FROM read_parquet('{base}') AS b
        JOIN read_parquet('{raw}') AS i USING (event_id)
        LEFT JOIN read_parquet('{specialist}') AS s USING (event_id)
        """
    )
    audit = connection.execute(
        f"""
        WITH base_ids AS (
            SELECT event_id FROM read_parquet('{base}')
        ), raw_ids AS (
            SELECT event_id FROM read_parquet('{raw}')
        ), specialist_ids AS (
            SELECT event_id FROM read_parquet('{specialist}')
        )
        SELECT
            (SELECT COUNT(*) FROM base_ids) AS base_rows,
            (SELECT COUNT(DISTINCT event_id) FROM base_ids) AS distinct_base_ids,
            (SELECT COUNT(*) FROM raw_ids) AS raw_rows,
            (SELECT COUNT(DISTINCT event_id) FROM raw_ids) AS distinct_raw_ids,
            (SELECT COUNT(*) FROM specialist_ids) AS specialist_rows,
            (SELECT COUNT(DISTINCT event_id) FROM specialist_ids)
                AS distinct_specialist_ids,
            (SELECT COUNT(*) FROM raw_ids ANTI JOIN base_ids USING (event_id))
                AS raw_ids_missing_from_base,
            (SELECT COUNT(*) FROM base_ids ANTI JOIN raw_ids USING (event_id))
                AS extra_base_ids,
            (SELECT COUNT(*) FROM specialist_ids ANTI JOIN base_ids USING (event_id))
                AS specialist_ids_missing_from_base
        """
    ).fetchone()
    names = [
        "base_rows",
        "distinct_base_ids",
        "raw_rows",
        "distinct_raw_ids",
        "specialist_rows",
        "distinct_specialist_ids",
        "raw_ids_missing_from_base",
        "extra_base_ids",
        "specialist_ids_missing_from_base",
    ]
    audit_result = dict(zip(names, (int(value) for value in audit)))
    if audit_result["base_rows"] != audit_result["distinct_base_ids"]:
        raise ValueError("Base predictions contain duplicate event_id values")
    if audit_result["raw_rows"] != audit_result["distinct_raw_ids"]:
        raise ValueError("Raw input contains duplicate event_id values")
    if audit_result["specialist_rows"] != audit_result["distinct_specialist_ids"]:
        raise ValueError("Specialist predictions contain duplicate event_id values")
    for key in (
        "raw_ids_missing_from_base",
        "extra_base_ids",
        "specialist_ids_missing_from_base",
    ):
        if audit_result[key]:
            raise ValueError(f"Submission event_id audit failed: {key}")

    label_rows = connection.execute(
        """
        SELECT pred_label, COUNT(*) AS rows
        FROM final_predictions GROUP BY pred_label ORDER BY pred_label
        """
    ).fetchall()
    label_counts = {str(label): int(rows) for label, rows in label_rows}
    unknown_labels = set(label_counts) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown prediction labels: {sorted(unknown_labels)}")
    connection.execute(
        f"""
        COPY (SELECT event_id, pred_label FROM final_predictions)
        TO '{output}' (HEADER, DELIMITER ',')
        """
    )
    connection.close()
    summary = {
        "output": str(args.output.resolve()),
        "rows": audit_result["raw_rows"],
        "label_counts": label_counts,
        "event_id_audit": audit_result,
        "size_mb": round(args.output.stat().st_size / 1024**2, 2),
        "suspicious_rules_enabled": not args.disable_suspicious_rules,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
