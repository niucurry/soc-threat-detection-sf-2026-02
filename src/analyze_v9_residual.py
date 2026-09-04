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
        description="Audit anchored evidence conflicts and changes from a V7 anchor"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evidence-source", choices=("metadata", "content"), default="metadata"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.predictions, args.features):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_column = f"{args.evidence_source}_reliability_candidate"
    connection = duckdb.connect()
    prediction_columns = {
        str(row[0])
        for row in connection.execute(
            f"""
            DESCRIBE SELECT *
            FROM read_parquet('{sql_path(args.predictions)}')
            """
        ).fetchall()
    }
    candidate_expressions: list[str] = []
    if "metadata_reliability_candidate" not in prediction_columns:
        candidate_expressions.append(
            "CAST(p.metadata_threat_probability > 0.5 AS TINYINT) "
            "AS metadata_reliability_candidate"
        )
    if "content_reliability_candidate" not in prediction_columns:
        candidate_expressions.append(
            "CAST(p.content_threat_probability > 0.5 AS TINYINT) "
            "AS content_reliability_candidate"
        )
    extra_candidates = (
        ",\n            " + ",\n            ".join(candidate_expressions)
        if candidate_expressions
        else ""
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW audited AS
        SELECT
            p.*{extra_candidates},
            f.pipeline,
            f.vendor_name,
            f.product_name,
            f.content_family,
            f.content_action,
            f.content_event_code,
            f.message_length_bucket
        FROM read_parquet('{sql_path(args.predictions)}') AS p
        INNER JOIN read_parquet('{sql_path(args.features)}') AS f USING (event_id)
        """
    )
    names = (
        "rows",
        "distinct_event_ids",
        "anchor_errors",
        "final_errors",
        "anchor_threat_errors",
        "final_threat_errors",
        "metadata_reliability_candidates",
        "metadata_candidate_true_threat",
        "metadata_candidate_true_benign",
        "content_reliability_candidates",
        "content_candidate_true_threat",
        "content_candidate_true_benign",
        "evidence_reliability_candidates",
        "evidence_candidate_true_threat",
        "evidence_candidate_true_benign",
        "conflict_candidates",
        "conflict_true_threat",
        "conflict_true_benign",
        "changed_predictions",
        "fixed_anchor_errors",
        "new_errors",
        "changed_threat_predictions",
        "fixed_anchor_threat_errors",
        "new_threat_errors",
    )
    row = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT event_id),
            SUM(anchor_pred_label <> true_label),
            SUM(pred_label <> true_label),
            SUM((anchor_pred_label = 'benign') <> (true_label = 'benign')),
            SUM((pred_label = 'benign') <> (true_label = 'benign')),
            SUM(metadata_reliability_candidate = 1),
            SUM(metadata_reliability_candidate = 1 AND true_label <> 'benign'),
            SUM(metadata_reliability_candidate = 1 AND true_label = 'benign'),
            SUM(content_reliability_candidate = 1),
            SUM(content_reliability_candidate = 1 AND true_label <> 'benign'),
            SUM(content_reliability_candidate = 1 AND true_label = 'benign'),
            SUM({evidence_column} = 1),
            SUM({evidence_column} = 1 AND true_label <> 'benign'),
            SUM({evidence_column} = 1 AND true_label = 'benign'),
            SUM(conflict_candidate = 1),
            SUM(conflict_candidate = 1 AND true_label <> 'benign'),
            SUM(conflict_candidate = 1 AND true_label = 'benign'),
            SUM(pred_label <> anchor_pred_label),
            SUM(anchor_pred_label <> true_label AND pred_label = true_label),
            SUM(anchor_pred_label = true_label AND pred_label <> true_label),
            SUM((anchor_pred_label = 'benign') <> (pred_label = 'benign')),
            SUM(
                (anchor_pred_label = 'benign') <> (true_label = 'benign')
                AND (pred_label = 'benign') = (true_label = 'benign')
            ),
            SUM(
                (anchor_pred_label = 'benign') = (true_label = 'benign')
                AND (pred_label = 'benign') <> (true_label = 'benign')
            )
        FROM audited
        """
    ).fetchone()
    summary: dict[str, Any] = {
        name: int(value or 0) for name, value in zip(names, row)
    }
    summary["evidence_source"] = args.evidence_source
    group_names = (
        "true_label",
        "anchor_pred_label",
        "pred_label",
        "vendor_name",
        "product_name",
        "content_family",
        "content_action",
        "rows",
        "min_anchor_threat",
        "avg_anchor_threat",
        "avg_metadata_threat",
        "avg_content_threat",
        "avg_final_threat",
        "avg_trust",
        "avg_delta",
    )
    groups = connection.execute(
        """
        SELECT
            true_label,
            anchor_pred_label,
            pred_label,
            vendor_name,
            product_name,
            content_family,
            content_action,
            COUNT(*) AS rows,
            MIN(anchor_threat_probability),
            AVG(anchor_threat_probability),
            AVG(metadata_threat_probability),
            AVG(content_threat_probability),
            AVG(threat_probability),
            AVG(trust_score),
            AVG(delta_margin)
        FROM audited
        WHERE conflict_candidate = 1
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        ORDER BY rows DESC, 1, 2, 3
        LIMIT 100
        """
    ).fetchall()
    summary["conflict_groups"] = [
        {name: value for name, value in zip(group_names, values)}
        for values in groups
    ]
    rows_path = args.output_dir / "conflict_rows.csv"
    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM audited
            WHERE conflict_candidate = 1 OR pred_label <> anchor_pred_label
            ORDER BY true_label, vendor_name, product_name, event_id
        ) TO '{sql_path(rows_path)}' (HEADER, DELIMITER ',')
        """
    )
    connection.close()
    if summary["rows"] != summary["distinct_event_ids"]:
        raise ValueError("Duplicate event_id values in V9 residual audit")
    summary["conflict_rows_csv"] = str(rows_path)
    output = args.output_dir / "residual_summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
