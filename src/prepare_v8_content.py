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

from soc_threat.content_features import DEFAULT_HASH_BUCKETS
from soc_threat.v8_content_features import (
    DEFAULT_V8_MESSAGE_CHARS_PER_VIEW,
    DEFAULT_V8_TOKENS_PER_VIEW,
    V8_CONTENT_VIEWS,
    V8ContentEncoding,
    encode_multiview_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = ["event_id", "message_sanitized"]


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _fixed_uint16(values: list[tuple[int, ...]], size: int) -> pa.FixedSizeListArray:
    flat = np.asarray(values, dtype=np.uint16).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.uint16()), size)


def _encoding_table(
    event_ids: list[str],
    encodings: list[V8ContentEncoding],
    *,
    tokens_per_view: int,
) -> pa.Table:
    width = len(V8_CONTENT_VIEWS) * tokens_per_view
    return pa.table(
        {
            "event_id": pa.array(event_ids, type=pa.string()),
            "multiview_token_ids": _fixed_uint16(
                [value.multiview_token_ids for value in encodings], width
            ),
            "head_token_count": pa.array(
                [value.head_token_count for value in encodings], type=pa.int16()
            ),
            "middle_token_count": pa.array(
                [value.middle_token_count for value in encodings], type=pa.int16()
            ),
            "tail_token_count": pa.array(
                [value.tail_token_count for value in encodings], type=pa.int16()
            ),
            "key_value_token_count": pa.array(
                [value.key_value_token_count for value in encodings], type=pa.int16()
            ),
        }
    )


def _valid_shard(path: Path, expected_rows: int, expected_width: int) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            return False
        field = parquet.schema_arrow.field("multiview_token_ids")
        return bool(
            pa.types.is_fixed_size_list(field.type)
            and field.type.list_size == expected_width
        )
    except (KeyError, OSError, pa.ArrowException):
        return False


def prepare_shards(
    raw_path: Path,
    shard_dir: Path,
    *,
    batch_size: int,
    progress_every: int,
    max_rows: int | None,
    buckets: int,
    tokens_per_view: int,
    chars_per_view: int,
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(raw_path)
    expected_width = len(V8_CONTENT_VIEWS) * tokens_per_view
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
        if _valid_shard(shard_path, expected_rows, expected_width):
            reused_rows += expected_rows
        else:
            rows = batch.to_pylist()
            encodings = [
                encode_multiview_content(
                    str(raw.get("message_sanitized") or ""),
                    buckets=buckets,
                    tokens_per_view=tokens_per_view,
                    chars_per_view=chars_per_view,
                )
                for raw in rows
            ]
            table = _encoding_table(
                [str(raw.get("event_id") or "") for raw in rows],
                encodings,
                tokens_per_view=tokens_per_view,
            )
            temporary = shard_path.with_suffix(".tmp.parquet")
            temporary.unlink(missing_ok=True)
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
                        "stage": "prepare_v8_multiview_shards",
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


def joined_file_is_valid(path: Path, expected_rows: int, expected_width: int) -> bool:
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_rows:
            return False
        field = parquet.schema_arrow.field("multiview_token_ids")
        return bool(
            pa.types.is_fixed_size_list(field.type)
            and field.type.list_size == expected_width
        )
    except (KeyError, OSError, pa.ArrowException):
        return False


def join_v6_and_multiview(
    v6_path: Path,
    shard_dir: Path,
    output_path: Path,
    *,
    expected_rows: int,
    expected_width: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.is_file() and not force:
        if joined_file_is_valid(output_path, expected_rows, expected_width):
            return {
                "output": str(output_path),
                "rows": expected_rows,
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
                b.* EXCLUDE (raw_token_ids, field_token_ids),
                c.* EXCLUDE (event_id)
            FROM read_parquet('{sql_path(v6_path)}') AS b
            INNER JOIN read_parquet('{sql_path(shard_dir / '*.parquet')}') AS c
                USING (event_id)
        ) TO '{sql_path(temporary)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
    )
    audit = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT event_id),
            AVG(head_token_count),
            AVG(middle_token_count),
            AVG(tail_token_count),
            AVG(key_value_token_count)
        FROM read_parquet('{sql_path(temporary)}')
        """
    ).fetchone()
    connection.close()
    if audit[0] != expected_rows or audit[0] != audit[1]:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"V8 join mismatch for {output_path}: expected {expected_rows}, "
            f"got rows={audit[0]}, distinct={audit[1]}"
        )
    os.replace(temporary, output_path)
    return {
        "base": str(v6_path),
        "content_shards": str(shard_dir),
        "output": str(output_path),
        "rows": int(audit[0]),
        "reused": False,
        "average_tokens": dict(zip(V8_CONTENT_VIEWS, map(float, audit[2:]))),
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "seconds": round(time.perf_counter() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare resumable V8 head/middle/tail/key-value content views"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v6-feature-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v8",
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    parser.add_argument(
        "--tokens-per-view", type=int, default=DEFAULT_V8_TOKENS_PER_VIEW
    )
    parser.add_argument(
        "--chars-per-view", type=int, default=DEFAULT_V8_MESSAGE_CHARS_PER_VIEW
    )
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.progress_every < 1:
        raise ValueError("batch size and progress interval must be positive")
    if args.tokens_per_view < 16 or args.chars_per_view < 256:
        raise ValueError("V8 view limits are too small")
    raw_train = args.data_dir / "train.parquet"
    raw_valid = args.data_dir / "valid_input.parquet"
    v6_train = args.v6_feature_dir / "v6_train.parquet"
    v6_valid = args.v6_feature_dir / "v6_valid.parquet"
    for path in (raw_train, raw_valid, v6_train, v6_valid):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_shards = args.output_dir / "train_multiview_shards"
    valid_shards = args.output_dir / "valid_multiview_shards"
    train_output = args.output_dir / "v8_train.parquet"
    valid_output = args.output_dir / "v8_valid.parquet"
    if args.force:
        for directory in (train_shards, valid_shards):
            if directory.exists():
                shutil.rmtree(directory)
        train_output.unlink(missing_ok=True)
        valid_output.unlink(missing_ok=True)

    started = time.perf_counter()
    shard_summaries = [
        prepare_shards(
            raw_train,
            train_shards,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            max_rows=args.max_train_rows,
            buckets=args.hash_buckets,
            tokens_per_view=args.tokens_per_view,
            chars_per_view=args.chars_per_view,
        ),
        prepare_shards(
            raw_valid,
            valid_shards,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            max_rows=args.max_valid_rows,
            buckets=args.hash_buckets,
            tokens_per_view=args.tokens_per_view,
            chars_per_view=args.chars_per_view,
        ),
    ]
    width = len(V8_CONTENT_VIEWS) * args.tokens_per_view
    joined = [
        join_v6_and_multiview(
            v6_train,
            train_shards,
            train_output,
            expected_rows=shard_summaries[0]["rows"],
            expected_width=width,
            force=args.force,
        ),
        join_v6_and_multiview(
            v6_valid,
            valid_shards,
            valid_output,
            expected_rows=shard_summaries[1]["rows"],
            expected_width=width,
            force=args.force,
        ),
    ]
    summary = {
        "version": "v8.0",
        "method": "four-view fixed-hash content encoding without template IDs",
        "views": list(V8_CONTENT_VIEWS),
        "hash_buckets": args.hash_buckets,
        "tokens_per_view": args.tokens_per_view,
        "total_token_width": width,
        "chars_per_view": args.chars_per_view,
        "shards": shard_summaries,
        "joined": joined,
        "total_seconds": round(time.perf_counter() - started, 2),
        "leakage_guard": (
            "Feature extraction is stateless and label-free; no validation labels, "
            "template IDs, schema IDs, event-specific rules, or fitted vocabulary are used"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v8_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
