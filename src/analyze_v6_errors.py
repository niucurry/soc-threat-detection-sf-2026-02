from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one V6 content-model run")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v4-predictions", type=Path)
    parser.add_argument("--v5-predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def grouped(
    connection: duckdb.DuckDBPyConnection,
    columns: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    selected = ", ".join(columns)
    positions = ", ".join(str(index) for index in range(1, len(columns) + 1))
    rows = connection.execute(
        f"""
        SELECT {selected}, COUNT(*) AS errors
        FROM joined
        WHERE true_label <> pred_label
        GROUP BY {positions}
        ORDER BY errors DESC, {positions}
        LIMIT ?
        """,
        [top_k],
    ).fetchall()
    return [
        {
            **{name: row[index] for index, name in enumerate(columns)},
            "errors": int(row[-1]),
        }
        for row in rows
    ]


def compare_predictions(
    connection: duckdb.DuckDBPyConnection,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, int]:
    row = connection.execute(
        f"""
        WITH comparison AS (
            SELECT
                b.event_id,
                b.true_label,
                b.pred_label AS baseline_label,
                c.pred_label AS candidate_label
            FROM read_parquet('{sql_path(baseline_path)}') AS b
            INNER JOIN read_parquet('{sql_path(candidate_path)}') AS c USING (event_id)
        )
        SELECT
            COUNT(*) AS rows,
            SUM(baseline_label <> true_label) AS baseline_errors,
            SUM(candidate_label <> true_label) AS candidate_errors,
            SUM(baseline_label <> true_label AND candidate_label = true_label)
                AS fixed_by_candidate,
            SUM(baseline_label = true_label AND candidate_label <> true_label)
                AS new_in_candidate,
            SUM(baseline_label <> true_label AND candidate_label <> true_label)
                AS both_wrong,
            SUM(baseline_label <> candidate_label) AS changed_predictions
        FROM comparison
        """
    ).fetchone()
    return {
        "rows": int(row[0]),
        "baseline_errors": int(row[1] or 0),
        "candidate_errors": int(row[2] or 0),
        "fixed_by_candidate": int(row[3] or 0),
        "new_in_candidate": int(row[4] or 0),
        "both_wrong": int(row[5] or 0),
        "changed_predictions": int(row[6] or 0),
    }


def main() -> None:
    args = parse_args()
    required = [args.predictions, args.features]
    for optional in (args.v4_predictions, args.v5_predictions):
        if optional is not None:
            required.append(optional)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE TEMP VIEW joined AS
        SELECT p.*, f.* EXCLUDE (event_id, label_binary)
        FROM read_parquet('{sql_path(args.predictions)}') AS p
        INNER JOIN read_parquet('{sql_path(args.features)}') AS f USING (event_id)
        """
    )
    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT event_id) AS distinct_ids,
            SUM(true_label <> pred_label) AS errors
        FROM joined
        """
    ).fetchone()
    if counts[0] != counts[1]:
        raise ValueError("Duplicate event_id values in V6 audit join")

    error_csv = args.output_dir / "error_rows.csv"
    connection.execute(
        f"""
        COPY (
            SELECT
                event_id,
                true_label,
                pred_label,
                prob_benign,
                prob_malicious,
                prob_suspicious,
                content_pred_label,
                pipeline,
                vendor_name,
                product_name,
                product_group,
                content_family,
                content_action,
                content_event_code,
                content_protocol,
                content_has_threat,
                content_has_authentication,
                content_has_potentially_harmful,
                raw_token_count,
                field_token_count,
                message_length_bucket,
                message_has_deny,
                message_has_failed,
                message_has_blocked
            FROM joined
            WHERE true_label <> pred_label
            ORDER BY true_label, pred_label, event_id
        ) TO '{sql_path(error_csv)}' (HEADER, DELIMITER ',')
        """
    )
    summary: dict[str, Any] = {
        "rows": int(counts[0]),
        "distinct_event_ids": int(counts[1]),
        "errors": int(counts[2] or 0),
        "error_rows_csv": str(error_csv),
        "by_confusion": grouped(connection, ("true_label", "pred_label"), args.top_k),
        "by_content_family": grouped(
            connection,
            ("true_label", "pred_label", "content_family", "content_action"),
            args.top_k,
        ),
        "by_security_signal": grouped(
            connection,
            (
                "true_label",
                "pred_label",
                "content_has_threat",
                "content_has_authentication",
                "content_has_potentially_harmful",
            ),
            args.top_k,
        ),
        "by_product": grouped(
            connection,
            ("true_label", "pred_label", "vendor_name", "product_name"),
            args.top_k,
        ),
    }
    if args.v4_predictions is not None:
        summary["v4_comparison"] = compare_predictions(
            connection, args.v4_predictions, args.predictions
        )
    if args.v5_predictions is not None:
        summary["v5_comparison"] = compare_predictions(
            connection, args.v5_predictions, args.predictions
        )
    connection.close()
    summary_path = args.output_dir / "error_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
