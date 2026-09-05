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
        description="Audit SOC train/validation overlap and distribution shift"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--valid-answer", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v2_2_generalization/audit.json"),
    )
    return parser.parse_args()


def fetch_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def overlap_summary(
    connection: duckdb.DuckDBPyConnection,
    mapping_table: str,
    value_expression: str,
    where: str = "true",
) -> list[dict[str, Any]]:
    return fetch_dicts(
        connection,
        f"""
        SELECT
            v.label_binary,
            COUNT(*) AS valid_rows,
            COUNT(m.group_value) AS seen_rows,
            SUM(CASE WHEN m.label_count = 1 THEN 1 ELSE 0 END)
                AS unambiguous_seen_rows,
            SUM(CASE WHEN m.majority_label = v.label_binary THEN 1 ELSE 0 END)
                AS majority_correct_rows
        FROM valid AS v
        LEFT JOIN {mapping_table} AS m
          ON {value_expression} = m.group_value
        WHERE {where}
        GROUP BY v.label_binary
        ORDER BY v.label_binary
        """,
    )


def main() -> None:
    args = parse_args()
    for path in (args.train, args.valid_input, args.valid_answer):
        if not path.is_file():
            raise FileNotFoundError(path)

    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        f"""
        CREATE VIEW train AS
        SELECT * FROM read_parquet('{sql_path(args.train)}')
        """
    )
    connection.execute(
        f"""
        CREATE VIEW valid AS
        SELECT i.*, a.label_binary
        FROM read_parquet('{sql_path(args.valid_input)}') AS i
        JOIN read_parquet('{sql_path(args.valid_answer)}') AS a USING (event_id)
        """
    )

    report: dict[str, Any] = {
        "inputs": {
            "train": str(args.train.resolve()),
            "valid_input": str(args.valid_input.resolve()),
            "valid_answer": str(args.valid_answer.resolve()),
        }
    }
    report["label_time_train"] = fetch_dicts(
        connection,
        """
        SELECT
            label_binary,
            COUNT(*) AS rows,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            COUNT(DISTINCT pipeline) AS pipelines,
            COUNT(DISTINCT COALESCE(product_name, '')) AS products,
            COUNT(DISTINCT COALESCE(message_sanitized, '')) AS messages
        FROM train GROUP BY label_binary ORDER BY label_binary
        """,
    )
    report["label_time_valid"] = fetch_dicts(
        connection,
        """
        SELECT
            label_binary,
            COUNT(*) AS rows,
            MIN(timestamp) AS min_timestamp,
            MAX(timestamp) AS max_timestamp,
            COUNT(DISTINCT pipeline) AS pipelines,
            COUNT(DISTINCT COALESCE(product_name, '')) AS products,
            COUNT(DISTINCT COALESCE(message_sanitized, '')) AS messages
        FROM valid GROUP BY label_binary ORDER BY label_binary
        """,
    )
    route_condition = "pipeline = 'syslog' AND COALESCE(product_name, '') = ''"
    report["specialist_route_train"] = fetch_dicts(
        connection,
        f"""
        SELECT
            label_binary,
            COUNT(*) AS rows,
            COUNT(DISTINCT COALESCE(message_sanitized, '')) AS messages
        FROM train WHERE {route_condition}
        GROUP BY label_binary ORDER BY label_binary
        """,
    )
    report["specialist_route_valid"] = fetch_dicts(
        connection,
        f"""
        SELECT
            label_binary,
            COUNT(*) AS rows,
            COUNT(DISTINCT COALESCE(message_sanitized, '')) AS messages
        FROM valid WHERE {route_condition}
        GROUP BY label_binary ORDER BY label_binary
        """,
    )
    report["unseen_valid_products"] = fetch_dicts(
        connection,
        """
        WITH train_products AS (
            SELECT DISTINCT COALESCE(product_name, '') AS product FROM train
        )
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT COALESCE(v.product_name, '')) AS products
        FROM valid AS v
        ANTI JOIN train_products AS t
          ON COALESCE(v.product_name, '') = t.product
        """,
    )[0]
    report["unseen_valid_pipelines"] = fetch_dicts(
        connection,
        """
        WITH train_pipelines AS (SELECT DISTINCT pipeline FROM train)
        SELECT COUNT(*) AS rows, COUNT(DISTINCT v.pipeline) AS pipelines
        FROM valid AS v
        ANTI JOIN train_pipelines AS t ON v.pipeline = t.pipeline
        """,
    )[0]
    report["valid_top_product_labels"] = fetch_dicts(
        connection,
        """
        SELECT
            COALESCE(product_name, '') AS product,
            label_binary,
            COUNT(*) AS rows
        FROM valid GROUP BY product, label_binary
        ORDER BY rows DESC LIMIT 30
        """,
    )

    exact_expression = "COALESCE(message_sanitized, '')"
    template_expression = """
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(COALESCE(message_sanitized, '')),
                '[0-9a-f]{16,}', '<hex>', 'g'
            ),
            '[0-9]+', '<n>', 'g'
        )
    """
    family_expression = f"SUBSTR(({template_expression}), 1, 48)"
    for name, expression in (
        ("exact_message", exact_expression),
        ("normalized_template", template_expression),
        ("normalized_family", family_expression),
    ):
        connection.execute(
            f"""
            CREATE TEMP TABLE {name}_counts AS
            SELECT
                {expression} AS group_value,
                label_binary,
                COUNT(*) AS rows
            FROM train
            GROUP BY group_value, label_binary
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE {name}_map AS
            SELECT
                group_value,
                ARG_MAX(label_binary, rows) AS majority_label,
                SUM(rows) AS train_rows,
                COUNT(*) AS label_count
            FROM {name}_counts
            GROUP BY group_value
            """
        )
        report[f"{name}_overlap_by_label"] = overlap_summary(
            connection,
            f"{name}_map",
            expression.replace("message_sanitized", "v.message_sanitized"),
        )
        report[f"{name}_route_overlap_by_label"] = overlap_summary(
            connection,
            f"{name}_map",
            expression.replace("message_sanitized", "v.message_sanitized"),
            "v.pipeline = 'syslog' AND COALESCE(v.product_name, '') = ''",
        )
        report[f"{name}_train_conflicts"] = fetch_dicts(
            connection,
            f"""
            SELECT
                COUNT(*) AS conflicting_groups,
                SUM(train_rows) AS rows_in_conflicting_groups
            FROM {name}_map WHERE label_count > 1
            """,
        )[0]

    strong_family = """
        CASE
            WHEN CONTAINS(LOWER(COALESCE(message_sanitized, '')), 'reject ok')
                THEN 'reject_ok'
            WHEN CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), '\"code\":\"4625\"'
            ) THEN 'windows_4625'
            WHEN STARTS_WITH(
                LOWER(COALESCE(message_sanitized, '')), 'org-1780 ::: tags='
            ) THEN 'org_tags'
            WHEN STARTS_WITH(
                LOWER(COALESCE(message_sanitized, '')), 'org-1780 ::: fqdn='
            ) AND CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), '=blocked'
            ) THEN 'org_fqdn_blocked'
            WHEN CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), ',traffic,deny,'
            ) THEN 'traffic_deny'
            WHEN CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), ' deny '
            ) THEN 'deny_word'
            ELSE 'other'
        END
    """
    existing_strong_rule = """
        CONTAINS(LOWER(COALESCE(message_sanitized, '')), 'reject ok')
        OR CONTAINS(
            LOWER(COALESCE(message_sanitized, '')), '\"code\":\"4625\"'
        )
        OR STARTS_WITH(
            LOWER(COALESCE(message_sanitized, '')), 'org-1780 ::: tags='
        )
        OR (
            STARTS_WITH(
                LOWER(COALESCE(message_sanitized, '')), 'org-1780 ::: fqdn='
            )
            AND CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), '=blocked'
            )
        )
        OR CONTAINS(LOWER(COALESCE(message_sanitized, '')), ' deny ')
        OR CONTAINS(
            LOWER(COALESCE(message_sanitized, '')), ',traffic,deny,'
        )
    """
    candidate_rules = {
        "traffic_drop": "CONTAINS(message, ',traffic,drop,')",
        "threat_block_url": (
            "CONTAINS(message, ',threat,url,') "
            "AND CONTAINS(message, 'block-url')"
        ),
        "protocol_deny": (
            "CONTAINS(message, 'tcp,deny') "
            "OR CONTAINS(message, 'udp,deny') "
            "OR CONTAINS(message, 'icmp,deny')"
        ),
        "comma_deny": "CONTAINS(message, ',deny,')",
        "decision_blocked": "CONTAINS(message, 'decision=blocked')",
        "act_deny": "CONTAINS(message, 'act=deny')",
        "comma_drop": "CONTAINS(message, ',drop,')",
    }
    flag_columns = ",\n".join(
        f"({expression}) AS {name}"
        for name, expression in candidate_rules.items()
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE candidate_rule_flags AS
        WITH combined AS (
            SELECT
                'train' AS split,
                label_binary,
                pipeline = 'syslog'
                    AND COALESCE(product_name, '') = '' AS on_route,
                LOWER(COALESCE(message_sanitized, '')) AS message,
                {existing_strong_rule} AS existing_rule
            FROM train
            UNION ALL
            SELECT
                'valid' AS split,
                label_binary,
                pipeline = 'syslog'
                    AND COALESCE(product_name, '') = '' AS on_route,
                LOWER(COALESCE(message_sanitized, '')) AS message,
                {existing_strong_rule} AS existing_rule
            FROM valid
        )
        SELECT split, label_binary, on_route, existing_rule, {flag_columns}
        FROM combined
        """
    )
    report["candidate_rule_audit"] = {}
    for name in candidate_rules:
        report["candidate_rule_audit"][name] = fetch_dicts(
            connection,
            f"""
            SELECT
                split,
                label_binary,
                COUNT(*) FILTER (WHERE {name}) AS rows,
                COUNT(*) FILTER (WHERE on_route AND {name}) AS route_rows,
                COUNT(*) FILTER (
                    WHERE on_route AND {name} AND NOT existing_rule
                ) AS incremental_route_rows
            FROM candidate_rule_flags
            GROUP BY split, label_binary
            HAVING COUNT(*) FILTER (WHERE {name}) > 0
            ORDER BY split, label_binary
            """,
        )
    expanded_valid_rule = f"""
        ({existing_strong_rule})
        OR CONTAINS(
            LOWER(COALESCE(message_sanitized, '')), ',traffic,drop,'
        )
        OR (
            CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), ',threat,url,'
            )
            AND CONTAINS(
                LOWER(COALESCE(message_sanitized, '')), 'block-url'
            )
        )
    """
    report["expanded_rule_valid_counts"] = fetch_dicts(
        connection,
        f"""
        SELECT label_binary, COUNT(*) AS rows
        FROM valid
        WHERE {route_condition} AND ({expanded_valid_rule})
        GROUP BY label_binary ORDER BY label_binary
        """,
    )
    report["expanded_rule_remaining_valid_malicious"] = fetch_dicts(
        connection,
        f"""
        SELECT event_id, SUBSTR(message_sanitized, 1, 800) AS message
        FROM valid
        WHERE {route_condition}
          AND label_binary = 'malicious'
          AND NOT ({expanded_valid_rule})
        LIMIT 20
        """,
    )
    report["specialist_semantic_families_train"] = fetch_dicts(
        connection,
        f"""
        SELECT
            label_binary,
            {strong_family} AS family,
            COUNT(*) AS rows,
            COUNT(DISTINCT {family_expression}) AS normalized_families
        FROM train WHERE {route_condition}
        GROUP BY label_binary, family ORDER BY label_binary, rows DESC
        """,
    )
    report["specialist_semantic_families_valid"] = fetch_dicts(
        connection,
        f"""
        SELECT
            label_binary,
            {strong_family} AS family,
            COUNT(*) AS rows,
            COUNT(DISTINCT {family_expression}) AS normalized_families
        FROM valid WHERE {route_condition}
        GROUP BY label_binary, family ORDER BY label_binary, rows DESC
        """,
    )
    report["top_malicious_families_train"] = fetch_dicts(
        connection,
        f"""
        SELECT {family_expression} AS family, COUNT(*) AS rows
        FROM train
        WHERE {route_condition} AND label_binary = 'malicious'
        GROUP BY family ORDER BY rows DESC LIMIT 30
        """,
    )
    report["top_malicious_families_valid"] = fetch_dicts(
        connection,
        f"""
        SELECT {family_expression} AS family, COUNT(*) AS rows
        FROM valid
        WHERE {route_condition} AND label_binary = 'malicious'
        GROUP BY family ORDER BY rows DESC LIMIT 30
        """,
    )

    connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
