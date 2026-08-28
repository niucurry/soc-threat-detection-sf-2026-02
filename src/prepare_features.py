from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("SOC_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))
)


FEATURE_QUERY = r"""
WITH base AS (
    SELECT
        s.event_id,
        s.label_binary,
        COALESCE(NULLIF(TRIM(s.pipeline), ''), '__MISSING__') AS pipeline,
        COALESCE(NULLIF(TRIM(s.product_name), ''), '__MISSING__') AS product_name,
        COALESCE(NULLIF(TRIM(s.vendor_name), ''), '__MISSING__') AS vendor_name,
        COALESCE(s.message_sanitized, '') AS message_text,
        LOWER(COALESCE(s.message_sanitized, '')) AS message_lower,
        TRY_CAST(s.src_port AS INTEGER) AS src_port_number,
        COALESCE(s.src_ip, '') AS src_ip_text,
        COALESCE(s.dst_ip, '') AS dst_ip_text,
        COALESCE(s.src_host, '') AS src_host_text,
        COALESCE(s.dst_host, '') AS dst_host_text,
        COALESCE(s.username, '') AS username_text,
        s.timestamp
    FROM source_data AS s
), engineered AS (
    SELECT
        *,
        CASE
            WHEN product_name = '__MISSING__' THEN 'missing'
            WHEN product_name = 'ASA Firewall' THEN 'asa'
            WHEN product_name = 'AWS VPC Security' THEN 'aws_vpc'
            WHEN product_name IN ('Precinct', 'Falcon') THEN 'other_suspicious_products'
            ELSE 'other'
        END AS product_group,
        CASE
            WHEN src_ip_text = '' THEN 'missing'
            WHEN regexp_full_match(src_ip_text, '[0-9]{1,3}(\\.[0-9]{1,3}){3}') THEN 'ipv4_shape'
            WHEN LOWER(src_ip_text) LIKE 'host-%' THEN 'host_token'
            ELSE 'other'
        END AS src_ip_kind,
        CASE
            WHEN src_port_number IS NULL THEN 'missing'
            WHEN src_port_number <= 1023 THEN '00000-01023'
            WHEN src_port_number <= 49151 THEN '01024-49151'
            ELSE '49152-65535'
        END AS port_bucket,
        CASE
            WHEN message_text = '' THEN 'missing'
            WHEN LENGTH(message_text) <= 120 THEN '001-120'
            WHEN LENGTH(message_text) <= 180 THEN '121-180'
            WHEN LENGTH(message_text) <= 300 THEN '181-300'
            WHEN LENGTH(message_text) <= 1000 THEN '301-1000'
            ELSE '1001+'
        END AS message_length_bucket,
        CAST(src_ip_text = '' AS TINYINT) AS src_ip_missing,
        CAST(dst_ip_text = '' AS TINYINT) AS dst_ip_missing,
        CAST(src_port_number IS NULL AS TINYINT) AS src_port_missing,
        CAST(src_host_text = '' AS TINYINT) AS src_host_missing,
        CAST(dst_host_text = '' AS TINYINT) AS dst_host_missing,
        CAST(username_text = '' AS TINYINT) AS username_missing,
        CAST(product_name = '__MISSING__' AS TINYINT) AS product_missing,
        CAST(message_text = '' AS TINYINT) AS message_missing,
        CAST((src_ip_text <> '') AS TINYINT)
          + CAST((dst_ip_text <> '') AS TINYINT)
          + CAST((src_port_number IS NOT NULL) AS TINYINT) AS network_present_count,
        LENGTH(message_text) AS message_length,
        LENGTH(src_ip_text) AS src_ip_length,
        LENGTH(dst_ip_text) AS dst_ip_length,
        LENGTH(src_host_text) AS src_host_length,
        LENGTH(dst_host_text) AS dst_host_length,
        LENGTH(username_text) AS username_length,
        CAST(message_lower LIKE '%deny%' AS TINYINT) AS message_has_deny,
        CAST(message_lower LIKE '%allow%' AS TINYINT) AS message_has_allow,
        CAST(message_lower LIKE '%accepted%' AS TINYINT) AS message_has_accepted,
        CAST(message_lower LIKE '%failed%' AS TINYINT) AS message_has_failed,
        CAST(message_lower LIKE '%blocked%' AS TINYINT) AS message_has_blocked,
        CAST(LEFT(LTRIM(message_text), 1) = '<' AS TINYINT) AS message_starts_angle,
        CAST(STRPOS(message_text, '{') > 0 AS TINYINT) AS message_contains_json,
        CAST(STRFTIME(TO_TIMESTAMP(timestamp), '%H') AS TINYINT) AS utc_hour,
        CAST(STRFTIME(TO_TIMESTAMP(timestamp), '%w') AS TINYINT) AS utc_weekday,
        STRFTIME(TO_TIMESTAMP(timestamp), '%Y-%m') AS year_month
    FROM base
)
SELECT
    event_id,
    label_binary,
    pipeline,
    product_name,
    vendor_name,
    product_group,
    src_ip_kind,
    port_bucket,
    message_length_bucket,
    pipeline || '|' || product_group || '|' || message_length_bucket || '|'
        || CASE WHEN message_has_deny = 1 THEN 'deny' ELSE 'not_deny' END
        AS structure_combo,
    CAST(src_ip_missing AS VARCHAR) || CAST(dst_ip_missing AS VARCHAR)
        || CAST(src_port_missing AS VARCHAR) AS network_missing_pattern,
    COALESCE(src_port_number, -1) AS src_port_number,
    src_ip_missing,
    dst_ip_missing,
    src_port_missing,
    src_host_missing,
    dst_host_missing,
    username_missing,
    product_missing,
    message_missing,
    network_present_count,
    message_length,
    src_ip_length,
    dst_ip_length,
    src_host_length,
    dst_host_length,
    username_length,
    message_has_deny,
    message_has_allow,
    message_has_accepted,
    message_has_failed,
    message_has_blocked,
    message_starts_angle,
    message_contains_json,
    utc_hour,
    utc_weekday,
    year_month
FROM engineered
"""


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def export_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    input_path: Path,
    output_path: Path,
    answer_path: Path | None,
    force: bool,
) -> dict[str, object]:
    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"{output_path} already exists; pass --force to replace it"
            )
        output_path.unlink()

    if answer_path is None:
        source_sql = f"SELECT * FROM read_parquet('{sql_path(input_path)}')"
    else:
        source_sql = f"""
            SELECT i.*, a.label_binary
            FROM read_parquet('{sql_path(input_path)}') AS i
            INNER JOIN read_parquet('{sql_path(answer_path)}') AS a USING (event_id)
        """
    connection.execute(f"CREATE OR REPLACE TEMP VIEW source_data AS {source_sql}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY ({FEATURE_QUERY})
        TO '{sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    row_count = connection.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(output_path)}')"
    ).fetchone()[0]
    labels = connection.execute(
        f"""
        SELECT label_binary, COUNT(*) AS rows
        FROM read_parquet('{sql_path(output_path)}')
        GROUP BY label_binary ORDER BY rows DESC
        """
    ).fetchall()
    return {
        "input": str(input_path),
        "answer": str(answer_path) if answer_path else None,
        "output": str(output_path),
        "rows": int(row_count),
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "labels": [{"label": label, "rows": int(rows)} for label, rows in labels],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact V1 SOC features")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_input = args.data_dir / "train.parquet"
    valid_input = args.data_dir / "valid_input.parquet"
    valid_answer = args.data_dir / "valid_answer_private.parquet"
    for path in (train_input, valid_input, valid_answer):
        if not path.exists():
            raise FileNotFoundError(path)

    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    summaries = [
        export_features(
            connection,
            input_path=train_input,
            output_path=args.output_dir / "v1_train.parquet",
            answer_path=None,
            force=args.force,
        ),
        export_features(
            connection,
            input_path=valid_input,
            output_path=args.output_dir / "v1_valid.parquet",
            answer_path=valid_answer,
            force=args.force,
        ),
    ]
    connection.close()
    manifest_path = args.output_dir / "v1_manifest.json"
    manifest_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
