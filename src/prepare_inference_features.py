from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from prepare_features import FEATURE_QUERY, sql_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare compact v1.0 features for an unlabeled test parquet"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        if not args.force:
            raise FileExistsError(
                f"{args.output} already exists; pass --force to replace it"
            )
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        f"""
        CREATE VIEW source_data AS
        SELECT *, CAST(NULL AS VARCHAR) AS label_binary
        FROM read_parquet('{sql_path(args.input)}')
        """
    )
    connection.execute(
        f"""
        COPY ({FEATURE_QUERY})
        TO '{sql_path(args.output)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    summary = connection.execute(
        f"""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT event_id) AS distinct_event_ids
        FROM read_parquet('{sql_path(args.output)}')
        """
    ).fetchone()
    connection.close()
    if summary[0] != summary[1]:
        raise ValueError("Input contains duplicate event_id values")
    print(
        {
            "input": str(args.input.resolve()),
            "output": str(args.output.resolve()),
            "rows": int(summary[0]),
            "size_mb": round(args.output.stat().st_size / 1024**2, 2),
        }
    )


if __name__ == "__main__":
    main()
