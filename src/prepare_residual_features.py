from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join v3.0 raw tokens and v4.1 multi-view tokens for v5 residuals"
    )
    parser.add_argument("--content-feature-dir", type=Path, required=True)
    parser.add_argument("--multiview-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def valid_output(
    path: Path,
    *,
    expected_rows: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        return (
            parquet.metadata.num_rows == expected_rows
            and {"event_id", "raw_token_ids", "multiview_token_ids"} <= names
        )
    except (OSError, KeyError):
        return False


def join_split(
    content_path: Path,
    multiview_path: Path,
    output_path: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    for path in (content_path, multiview_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    content_rows = row_count(content_path)
    multiview_rows = row_count(multiview_path)
    if content_rows != multiview_rows:
        raise ValueError(
            "Content/multi-view row mismatch: "
            f"{content_path}={content_rows}, {multiview_path}={multiview_rows}"
        )
    if not force and valid_output(output_path, expected_rows=multiview_rows):
        return {
            "output": str(output_path),
            "rows": multiview_rows,
            "reused": True,
            "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY (
            SELECT
                m.*,
                r.raw_token_ids
            FROM read_parquet('{sql_path(multiview_path)}') AS m
            INNER JOIN read_parquet('{sql_path(content_path)}') AS r USING (event_id)
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    rows, distinct_ids = connection.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT event_id)
        FROM read_parquet('{sql_path(temporary)}')
        """
    ).fetchone()
    connection.close()
    if rows != multiview_rows or rows != distinct_ids:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"v5 residual join mismatch for {output_path}: expected={multiview_rows}, "
            f"rows={rows}, distinct_event_ids={distinct_ids}"
        )
    os.replace(temporary, output_path)
    return {
        "content": str(content_path),
        "multiview": str(multiview_path),
        "output": str(output_path),
        "rows": int(rows),
        "distinct_event_ids": int(distinct_ids),
        "reused": False,
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    summaries = []
    for split in ("train", "valid"):
        summaries.append(
            join_split(
                args.content_feature_dir / f"content_{split}.parquet",
                args.multiview_feature_dir / f"multiview_{split}.parquet",
                args.output_dir / f"residual_{split}.parquet",
                force=args.force,
            )
        )
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    manifest = {
        "model_version": "v5.x",
        "method": "event_id join of v3.0 raw and v4.1 multi-view inputs",
        "splits": summaries,
        "total_seconds": round(time.perf_counter() - started, 2),
        "leakage_guard": (
            "The join uses event_id only and adds no labels or validation-fitted state"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "residual_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
