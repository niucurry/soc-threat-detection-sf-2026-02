from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export V5.1 neural-base errors and optional V4/V5 comparison"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v4-predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def grouped_errors(
    connection: duckdb.DuckDBPyConnection,
    *,
    columns: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    selected = ", ".join(columns)
    grouped = ", ".join(str(index) for index in range(1, len(columns) + 1))
    rows = connection.execute(
        f"""
        SELECT {selected}, COUNT(*) AS errors
        FROM v5_joined
        WHERE true_label <> pred_label
        GROUP BY {grouped}
        ORDER BY errors DESC, {grouped}
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


def main() -> None:
    args = parse_args()
    required = [args.predictions, args.features]
    if args.v4_predictions is not None:
        required.append(args.v4_predictions)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE TEMP VIEW v5_joined AS
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
            SUM(CASE WHEN true_label <> pred_label THEN 1 ELSE 0 END) AS errors
        FROM v5_joined
        """
    ).fetchone()
    if counts[0] != counts[1]:
        raise ValueError("Duplicate event_id values after prediction/feature join")

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
                pipeline,
                vendor_name,
                product_name,
                message_format,
                parser_type,
                structured_parser,
                payload_parse_status,
                template_id,
                template_seen_train,
                schema_id,
                schema_seen_train,
                semantic_template_id,
                semantic_template_seen_train,
                event_category_v5,
                event_type_v5,
                event_action_v5,
                event_outcome_v5,
                event_reason_v5,
                authentication_factor,
                service_name_v5,
                application_name_v5,
                rule_name_v5,
                threat_category_v5,
                network_protocol,
                event_code,
                dst_port_bucket,
                http_method,
                event_severity_number,
                malware_present,
                authentication_present
            FROM v5_joined
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
        "by_confusion": grouped_errors(
            connection,
            columns=("true_label", "pred_label"),
            top_k=args.top_k,
        ),
        "by_format": grouped_errors(
            connection,
            columns=(
                "true_label",
                "pred_label",
                "message_format",
                "payload_parse_status",
            ),
            top_k=args.top_k,
        ),
        "by_semantics": grouped_errors(
            connection,
            columns=(
                "true_label",
                "pred_label",
                "event_category_v5",
                "event_action_v5",
                "event_code",
            ),
            top_k=args.top_k,
        ),
        "by_seen_state": grouped_errors(
            connection,
            columns=(
                "true_label",
                "pred_label",
                "template_seen_train",
                "schema_seen_train",
                "semantic_template_seen_train",
            ),
            top_k=args.top_k,
        ),
    }

    if args.v4_predictions is not None:
        connection.execute(
            f"""
            CREATE TEMP VIEW version_comparison AS
            SELECT
                v4.event_id,
                v4.true_label,
                v4.pred_label AS v4_pred_label,
                v5.pred_label AS v5_pred_label
            FROM read_parquet('{sql_path(args.v4_predictions)}') AS v4
            INNER JOIN read_parquet('{sql_path(args.predictions)}') AS v5
                USING (event_id)
            """
        )
        comparison = connection.execute(
            """
            SELECT
                COUNT(*) AS rows,
                SUM(v4_pred_label <> true_label) AS v4_errors,
                SUM(v5_pred_label <> true_label) AS v5_errors,
                SUM(v4_pred_label <> true_label AND v5_pred_label = true_label)
                    AS v4_wrong_v5_correct,
                SUM(v4_pred_label = true_label AND v5_pred_label <> true_label)
                    AS v4_correct_v5_wrong,
                SUM(v4_pred_label <> true_label AND v5_pred_label <> true_label)
                    AS both_wrong,
                SUM(v4_pred_label <> v5_pred_label) AS changed_predictions
            FROM version_comparison
            """
        ).fetchone()
        comparison_summary = {
            "rows": int(comparison[0]),
            "v4_errors": int(comparison[1] or 0),
            "v5_errors": int(comparison[2] or 0),
            "v4_wrong_v5_correct": int(comparison[3] or 0),
            "v4_correct_v5_wrong": int(comparison[4] or 0),
            "both_wrong": int(comparison[5] or 0),
            "changed_predictions": int(comparison[6] or 0),
        }
        summary["v4_v5_comparison"] = comparison_summary

    connection.close()
    summary_path = args.output_dir / "error_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
