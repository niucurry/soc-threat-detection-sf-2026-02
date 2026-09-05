from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one v4 hierarchical run")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drain-predictions", type=Path)
    parser.add_argument("--structured-predictions", type=Path)
    parser.add_argument("--content-predictions", type=Path)
    parser.add_argument("--hierarchical-predictions", type=Path)
    parser.add_argument("--top-k", type=int, default=40)
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
        FROM audited
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
            SUM(baseline_label <> true_label AND candidate_label = true_label),
            SUM(baseline_label = true_label AND candidate_label <> true_label),
            SUM(baseline_label <> true_label AND candidate_label <> true_label),
            SUM(baseline_label <> candidate_label)
        FROM comparison
        """
    ).fetchone()
    names = [
        "rows",
        "baseline_errors",
        "candidate_errors",
        "fixed_by_candidate",
        "new_in_candidate",
        "both_wrong",
        "changed_predictions",
    ]
    return {name: int(value or 0) for name, value in zip(names, row)}


def main() -> None:
    args = parse_args()
    required = [args.predictions, args.features]
    optional = {
        "v1.1_drain_comparison": args.drain_predictions,
        "v1.2_structured_comparison": args.structured_predictions,
        "v3.0_exp02_comparison": args.content_predictions,
        "v4.0_exp02_comparison": args.hierarchical_predictions,
    }
    required.extend(path for path in optional.values() if path is not None)
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
    connection.execute(
        """
        CREATE TEMP VIEW audited AS
        SELECT
            *,
            CASE
                WHEN true_label = 'benign' AND pred_label <> 'benign'
                    THEN 'threat_false_positive'
                WHEN true_label <> 'benign' AND pred_label = 'benign'
                    THEN 'threat_false_negative'
                WHEN true_label <> pred_label THEN 'subtype_confusion'
                ELSE 'correct'
            END AS error_kind,
            CASE
                WHEN semantic_combo_count = 0 THEN 'unseen'
                WHEN semantic_combo_count < 10 THEN 'seen_001_009'
                WHEN semantic_combo_count < 100 THEN 'seen_010_099'
                WHEN semantic_combo_count < 1000 THEN 'seen_100_999'
                ELSE 'seen_1000_plus'
            END AS combo_frequency_bucket
        FROM joined
        """
    )
    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT event_id) AS distinct_ids,
            SUM(true_label <> pred_label) AS errors,
            SUM(error_kind = 'threat_false_negative') AS threat_false_negative,
            SUM(error_kind = 'threat_false_positive') AS threat_false_positive,
            SUM(error_kind = 'subtype_confusion') AS subtype_confusion,
            SUM(semantic_combo_count = 0) AS unseen_combo_rows,
            SUM(semantic_combo_count = 0 AND true_label <> pred_label)
                AS unseen_combo_errors
        FROM audited
        """
    ).fetchone()
    if counts[0] != counts[1]:
        raise ValueError("Duplicate event_id values in v4 audit join")

    error_csv = args.output_dir / "error_rows.csv"
    connection.execute(
        f"""
        COPY (
            SELECT
                event_id,
                true_label,
                pred_label,
                error_kind,
                prob_benign,
                prob_malicious,
                prob_suspicious,
                threat_probability,
                subtype_pred_label,
                metadata_subtype_pred_label,
                content_threat_probability,
                metadata_threat_probability,
                semantic_combo_count,
                novelty_gate,
                combo_frequency_bucket,
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
                message_length_bucket
            FROM audited
            WHERE true_label <> pred_label
            ORDER BY error_kind, true_label, pred_label, event_id
        ) TO '{sql_path(error_csv)}' (HEADER, DELIMITER ',')
        """
    )

    summary: dict[str, Any] = {
        "rows": int(counts[0]),
        "distinct_event_ids": int(counts[1]),
        "errors": int(counts[2] or 0),
        "threat_false_negative": int(counts[3] or 0),
        "threat_false_positive": int(counts[4] or 0),
        "subtype_confusion": int(counts[5] or 0),
        "unseen_combo_rows": int(counts[6] or 0),
        "unseen_combo_errors": int(counts[7] or 0),
        "error_rows_csv": str(error_csv),
        "by_error_kind": grouped(
            connection, ("error_kind", "true_label", "pred_label"), args.top_k
        ),
        "by_combo_frequency": grouped(
            connection,
            (
                "error_kind",
                "combo_frequency_bucket",
                "content_family",
                "content_action",
            ),
            args.top_k,
        ),
        "by_product": grouped(
            connection,
            (
                "error_kind",
                "true_label",
                "pred_label",
                "vendor_name",
                "product_name",
            ),
            args.top_k,
        ),
        "by_semantics": grouped(
            connection,
            (
                "error_kind",
                "true_label",
                "pred_label",
                "content_family",
                "content_action",
                "content_has_threat",
                "content_has_authentication",
            ),
            args.top_k,
        ),
    }
    for name, path in optional.items():
        if path is not None:
            summary[name] = compare_predictions(connection, path, args.predictions)
    connection.close()
    (args.output_dir / "error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
