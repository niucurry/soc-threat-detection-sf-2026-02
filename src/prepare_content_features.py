from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from soc_threat.content_features import (
    DEFAULT_HASH_BUCKETS,
    DEFAULT_MAX_TOKENS,
    ContentEncoding,
    encode_log_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = [
    "event_id",
    "pipeline",
    "product_name",
    "vendor_name",
    "message_sanitized",
]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _fixed_uint16(values: list[tuple[int, ...]], size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.uint16()), size)


def _encoding_table(
    event_ids: list[str], encodings: list[ContentEncoding], max_tokens: int
) -> pa.Table:
    return pa.table(
        {
            "event_id": pa.array(event_ids, type=pa.string()),
            "raw_token_ids": _fixed_uint16(
                [value.raw_token_ids for value in encodings], max_tokens
            ),
            "field_token_ids": _fixed_uint16(
                [value.field_token_ids for value in encodings], max_tokens
            ),
            "raw_token_count": pa.array(
                [value.raw_token_count for value in encodings], type=pa.int16()
            ),
            "field_token_count": pa.array(
                [value.field_token_count for value in encodings], type=pa.int16()
            ),
            "content_family": [value.content_family for value in encodings],
            "content_action": [value.content_action for value in encodings],
            "content_event_code": [value.content_event_code for value in encodings],
            "content_protocol": [value.content_protocol for value in encodings],
            "content_has_threat": pa.array(
                [value.content_has_threat for value in encodings], type=pa.int8()
            ),
            "content_has_authentication": pa.array(
                [value.content_has_authentication for value in encodings],
                type=pa.int8(),
            ),
            "content_has_potentially_harmful": pa.array(
                [value.content_has_potentially_harmful for value in encodings],
                type=pa.int8(),
            ),
        }
    )


def _valid_shard(path: Path, expected_rows: int, max_tokens: int) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            return False
        schema = parquet.schema_arrow
        for name in ("raw_token_ids", "field_token_ids"):
            field = schema.field(name)
            if not pa.types.is_fixed_size_list(field.type):
                return False
            if field.type.list_size != max_tokens:
                return False
        return True
    except (OSError, pa.ArrowException):
        return False


def prepare_shards(
    raw_path: Path,
    shard_dir: Path,
    *,
    batch_size: int,
    progress_every: int,
    max_rows: int | None,
    buckets: int,
    max_tokens: int,
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(raw_path)
    processed = 0
    reused_rows = 0
    written_rows = 0
    started = time.perf_counter()
    next_progress = progress_every
    for shard_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=RAW_COLUMNS)
    ):
        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
        expected_rows = len(batch)
        shard_path = shard_dir / f"part-{shard_index:06d}.parquet"
        if _valid_shard(shard_path, expected_rows, max_tokens):
            reused_rows += expected_rows
        else:
            rows = batch.to_pylist()
            encodings = [
                encode_log_content(raw, buckets=buckets, max_tokens=max_tokens)
                for raw in rows
            ]
            table = _encoding_table(
                [str(raw.get("event_id") or "") for raw in rows],
                encodings,
                max_tokens,
            )
            temporary = shard_path.with_suffix(".tmp.parquet")
            if temporary.exists():
                temporary.unlink()
            pq.write_table(
                table,
                temporary,
                compression="zstd",
                row_group_size=batch_size,
            )
            os.replace(temporary, shard_path)
            written_rows += expected_rows
        processed += expected_rows
        if processed >= next_progress:
            print(
                json.dumps(
                    {
                        "stage": "prepare_content_features_shards",
                        "input": raw_path.name,
                        "rows": processed,
                        "reused_rows": reused_rows,
                        "written_rows": written_rows,
                        "seconds": round(time.perf_counter() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_progress += progress_every
    if processed == 0:
        raise ValueError(f"No rows were read from {raw_path}")
    return {
        "input": str(raw_path),
        "shard_dir": str(shard_dir),
        "rows": processed,
        "reused_rows": reused_rows,
        "written_rows": written_rows,
        "seconds": round(time.perf_counter() - started, 2),
    }


def join_base_and_content(
    base_path: Path,
    shard_dir: Path,
    output_path: Path,
    *,
    expected_rows: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.is_file() and not force:
        try:
            rows = duckdb.sql(
                f"SELECT COUNT(*) FROM read_parquet('{sql_path(output_path)}')"
            ).fetchone()[0]
            if rows == expected_rows:
                return {
                    "output": str(output_path),
                    "rows": int(rows),
                    "reused": True,
                    "size_mb": round(output_path.stat().st_size / 1024**2, 2),
                }
        except duckdb.Error:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY (
            SELECT b.*, c.* EXCLUDE (event_id)
            FROM read_parquet('{sql_path(base_path)}') AS b
            INNER JOIN read_parquet('{sql_path(shard_dir / "*.parquet")}') AS c
                USING (event_id)
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    audit = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT event_id) AS distinct_ids,
            AVG(raw_token_count) AS average_raw_tokens,
            AVG(field_token_count) AS average_field_tokens
        FROM read_parquet('{sql_path(temporary)}')
        """
    ).fetchone()
    connection.close()
    if audit[0] != expected_rows or audit[0] != audit[1]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Content join mismatch for {output_path}: expected {expected_rows}, "
            f"got rows={audit[0]}, distinct={audit[1]}"
        )
    os.replace(temporary, output_path)
    return {
        "base": str(base_path),
        "content_shards": str(shard_dir),
        "output": str(output_path),
        "rows": int(audit[0]),
        "reused": False,
        "average_raw_tokens": float(audit[2]),
        "average_field_tokens": float(audit[3]),
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "seconds": round(time.perf_counter() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare resumable v3.0 raw and field-aware content hashes"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--base-feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v3_0",
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete only v3.0 content shards and joined files before rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.progress_every < 1 or args.max_tokens < 8:
        raise ValueError(
            "batch size, progress interval, and token limit must be positive"
        )
    raw_train = args.data_dir / "train.parquet"
    raw_valid = args.data_dir / "valid_input.parquet"
    base_train = args.base_feature_dir / "tabular_train.parquet"
    base_valid = args.base_feature_dir / "tabular_valid.parquet"
    for path in (raw_train, raw_valid, base_train, base_valid):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_shards = args.output_dir / "train_content_shards"
    valid_shards = args.output_dir / "valid_content_shards"
    train_output = args.output_dir / "content_train.parquet"
    valid_output = args.output_dir / "content_valid.parquet"
    if args.force:
        for directory in (train_shards, valid_shards):
            if directory.exists():
                shutil.rmtree(directory)
        for path in (train_output, valid_output):
            path.unlink(missing_ok=True)

    started = time.perf_counter()
    shard_summaries = [
        prepare_shards(
            raw_train,
            train_shards,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            max_rows=args.max_train_rows,
            buckets=args.hash_buckets,
            max_tokens=args.max_tokens,
        ),
        prepare_shards(
            raw_valid,
            valid_shards,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            max_rows=args.max_valid_rows,
            buckets=args.hash_buckets,
            max_tokens=args.max_tokens,
        ),
    ]
    joined = [
        join_base_and_content(
            base_train,
            train_shards,
            train_output,
            expected_rows=shard_summaries[0]["rows"],
            force=args.force,
        ),
        join_base_and_content(
            base_valid,
            valid_shards,
            valid_output,
            expected_rows=shard_summaries[1]["rows"],
            force=args.force,
        ),
    ]
    summary = {
        "model_version": "v3.0",
        "legacy_alias": "V6",
        "method": "fixed-hash word/bigram/character content encoding without template IDs",
        "hash_buckets": args.hash_buckets,
        "max_tokens": args.max_tokens,
        "shards": shard_summaries,
        "joined": joined,
        "total_seconds": round(time.perf_counter() - started, 2),
        "leakage_guard": (
            "Hashing is fixed and stateless; validation labels, vocabulary, Drain, schema IDs, "
            "and template IDs are not used by content preparation"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "content_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
