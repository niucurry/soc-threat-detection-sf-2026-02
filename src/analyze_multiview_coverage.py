from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb


VIEW_COUNT_COLUMNS = (
    "head_token_count",
    "middle_token_count",
    "tail_token_count",
    "key_value_token_count",
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v4.1 multi-view coverage and branch disagreement"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-view", type=int, default=64)
    return parser.parse_args()


def row_dict(names: tuple[str, ...], values: tuple[Any, ...]) -> dict[str, Any]:
    return {name: int(value or 0) for name, value in zip(names, values)}


def main() -> None:
    args = parse_args()
    for path in (args.predictions, args.features):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.tokens_per_view < 1:
        raise ValueError("tokens-per-view must be positive")

    connection = duckdb.connect()
    connection.execute(
        f"""
        CREATE TEMP VIEW audited AS
        SELECT
            p.*,
            f.head_token_count,
            f.middle_token_count,
            f.tail_token_count,
            f.key_value_token_count,
            f.content_family,
            f.content_action,
            f.vendor_name,
            f.product_name,
            f.message_length_bucket
        FROM read_parquet('{sql_path(args.predictions)}') AS p
        INNER JOIN read_parquet('{sql_path(args.features)}') AS f USING (event_id)
        """
    )
    names = (
        "rows",
        "distinct_event_ids",
        "errors",
        "all_views_full_rows",
        "error_all_views_full_rows",
        "final_benign_metadata_threat_rows",
        "final_benign_metadata_threat_true_threat",
        "final_benign_content_threat_rows",
        "final_benign_content_threat_true_threat",
    )
    values = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT event_id),
            SUM(true_label <> pred_label),
            SUM(head_token_count = ? AND middle_token_count = ?
                AND tail_token_count = ? AND key_value_token_count = ?),
            SUM(true_label <> pred_label AND head_token_count = ?
                AND middle_token_count = ? AND tail_token_count = ?
                AND key_value_token_count = ?),
            SUM(pred_label = 'benign' AND metadata_threat_probability >= 0.5),
            SUM(pred_label = 'benign' AND metadata_threat_probability >= 0.5
                AND true_label <> 'benign'),
            SUM(pred_label = 'benign' AND content_threat_probability >= 0.5),
            SUM(pred_label = 'benign' AND content_threat_probability >= 0.5
                AND true_label <> 'benign')
        FROM audited
        """,
        [args.tokens_per_view] * 8,
    ).fetchone()
    summary: dict[str, Any] = row_dict(names, values)
    summary["tokens_per_view"] = args.tokens_per_view
    summary["average_view_tokens"] = dict(
        zip(
            VIEW_COUNT_COLUMNS,
            map(
                float,
                connection.execute(
                    """
                    SELECT
                        AVG(head_token_count), AVG(middle_token_count),
                        AVG(tail_token_count), AVG(key_value_token_count)
                    FROM audited
                    """
                ).fetchone(),
            ),
        )
    )
    grouped = connection.execute(
        """
        SELECT
            true_label,
            pred_label,
            vendor_name,
            product_name,
            content_family,
            content_action,
            message_length_bucket,
            MIN(head_token_count),
            MIN(middle_token_count),
            MIN(tail_token_count),
            MIN(key_value_token_count),
            COUNT(*) AS errors
        FROM audited
        WHERE true_label <> pred_label
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        ORDER BY errors DESC, 1, 2
        LIMIT 50
        """
    ).fetchall()
    group_names = (
        "true_label",
        "pred_label",
        "vendor_name",
        "product_name",
        "content_family",
        "content_action",
        "message_length_bucket",
        "min_head_tokens",
        "min_middle_tokens",
        "min_tail_tokens",
        "min_key_value_tokens",
        "errors",
    )
    summary["error_groups"] = [
        {name: value for name, value in zip(group_names, row)} for row in grouped
    ]
    connection.close()
    if summary["rows"] != summary["distinct_event_ids"]:
        raise ValueError("Duplicate event_id values in v4.1 coverage join")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
