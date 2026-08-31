from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from soc_threat.log_semantics import DrainSettings, GroupedDrainModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("SOC_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))
)
RAW_COLUMNS = [
    "event_id",
    "pipeline",
    "product_name",
    "vendor_name",
    "message_sanitized",
]


def iter_raw_rows(
    path: Path,
    *,
    batch_size: int,
    max_rows: int | None,
) -> Iterator[list[dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    emitted = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=RAW_COLUMNS):
        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
        rows = batch.to_pylist()
        emitted += len(rows)
        yield rows


def fit_template_model(
    train_path: Path,
    *,
    model_dir: Path,
    batch_size: int,
    max_rows: int | None,
    progress_every: int,
    settings: DrainSettings,
    force: bool,
) -> dict[str, Any]:
    model = GroupedDrainModel(settings)
    processed = 0
    started = time.perf_counter()
    next_progress = progress_every
    for rows in iter_raw_rows(train_path, batch_size=batch_size, max_rows=max_rows):
        for raw in rows:
            model.fit_raw(raw)
        processed += len(rows)
        if processed >= next_progress:
            print(
                json.dumps(
                    {
                        "stage": "fit_templates",
                        "rows": processed,
                        "groups": len(model.miners),
                        "seconds": round(time.perf_counter() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            next_progress += progress_every
    model.save(model_dir, force=force)
    summary = {
        **model.summary(),
        "model_dir": str(model_dir),
        "fit_seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps({"stage": "template_model_ready", **summary}, ensure_ascii=False), flush=True)
    return summary


def write_log_features(
    input_path: Path,
    output_path: Path,
    *,
    model: GroupedDrainModel,
    batch_size: int,
    max_rows: int | None,
    progress_every: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"{output_path} exists; pass --force to replace it")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    processed = 0
    started = time.perf_counter()
    next_progress = progress_every
    try:
        for rows in iter_raw_rows(input_path, batch_size=batch_size, max_rows=max_rows):
            records = [model.feature_record(raw) for raw in rows]
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table, row_group_size=batch_size)
            processed += len(records)
            if processed >= next_progress:
                print(
                    json.dumps(
                        {
                            "stage": "write_log_features",
                            "input": input_path.name,
                            "rows": processed,
                            "seconds": round(time.perf_counter() - started, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_progress += progress_every
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"No rows were read from {input_path}")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": processed,
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "seconds": round(time.perf_counter() - started, 2),
    }


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def join_base_and_log_features(
    base_path: Path,
    log_path: Path,
    output_path: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists():
        if not force:
            raise FileExistsError(f"{output_path} exists; pass --force to replace it")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    connection.execute("SET preserve_insertion_order = false")
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY (
            SELECT b.*, l.* EXCLUDE (event_id)
            FROM read_parquet('{_sql_path(base_path)}') AS b
            INNER JOIN read_parquet('{_sql_path(log_path)}') AS l USING (event_id)
        )
        TO '{_sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    audit = connection.execute(
        f"""
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT event_id) AS distinct_event_ids,
               SUM(CASE WHEN template_seen_train = 0 THEN 1 ELSE 0 END) AS unseen_templates,
               SUM(CASE WHEN parser_type = 'drain_unmatched' THEN 1 ELSE 0 END) AS drain_unmatched
        FROM read_parquet('{_sql_path(output_path)}')
        """
    ).fetchone()
    connection.close()
    if audit[0] != audit[1]:
        raise ValueError(f"Duplicate event_id values in {output_path}")
    return {
        "base": str(base_path),
        "log": str(log_path),
        "output": str(output_path),
        "rows": int(audit[0]),
        "unseen_templates": int(audit[2] or 0),
        "drain_unmatched": int(audit[3] or 0),
        "size_mb": round(output_path.stat().st_size / 1024**2, 2),
        "seconds": round(time.perf_counter() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare V4 hybrid-parser and grouped-Drain neural features"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--base-feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v4",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v4_drain_neural" / "template_model",
    )
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--drain-sim-th", type=float, default=0.50)
    parser.add_argument("--drain-depth", type=int, default=4)
    parser.add_argument("--drain-max-children", type=int, default=100)
    parser.add_argument("--drain-max-clusters", type=int, default=5000)
    parser.add_argument("--max-message-chars", type=int, default=2048)
    parser.add_argument("--max-groups", type=int, default=128)
    parser.add_argument(
        "--reuse-template-model",
        action="store_true",
        help="Load --model-dir instead of fitting Drain templates again",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_raw = args.data_dir / "train.parquet"
    valid_raw = args.data_dir / "valid_input.parquet"
    base_train = args.base_feature_dir / "v1_train.parquet"
    base_valid = args.base_feature_dir / "v1_valid.parquet"
    for path in (train_raw, valid_raw, base_train, base_valid):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1 or args.progress_every < 1:
        raise ValueError("--batch-size and --progress-every must be positive")
    settings = DrainSettings(
        similarity_threshold=args.drain_sim_th,
        depth=args.drain_depth,
        max_children=args.drain_max_children,
        max_clusters_per_group=args.drain_max_clusters,
        max_message_chars=args.max_message_chars,
        max_groups=args.max_groups,
    )

    started = time.perf_counter()
    if args.reuse_template_model:
        model = GroupedDrainModel.load(args.model_dir)
        template_manifest = model.summary()
    else:
        template_manifest = fit_template_model(
            train_raw,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            max_rows=args.max_train_rows,
            progress_every=args.progress_every,
            settings=settings,
            force=args.force,
        )
        model = GroupedDrainModel.load(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_log = args.output_dir / "v4_train_log_features.parquet"
    valid_log = args.output_dir / "v4_valid_log_features.parquet"
    log_summaries = [
        write_log_features(
            train_raw,
            train_log,
            model=model,
            batch_size=args.batch_size,
            max_rows=args.max_train_rows,
            progress_every=args.progress_every,
            force=args.force,
        ),
        write_log_features(
            valid_raw,
            valid_log,
            model=model,
            batch_size=args.batch_size,
            max_rows=args.max_valid_rows,
            progress_every=args.progress_every,
            force=args.force,
        ),
    ]
    joined = [
        join_base_and_log_features(
            base_train,
            train_log,
            args.output_dir / "v4_train.parquet",
            force=args.force,
        ),
        join_base_and_log_features(
            base_valid,
            valid_log,
            args.output_dir / "v4_valid.parquet",
            force=args.force,
        ),
    ]
    summary = {
        "template_model": template_manifest,
        "log_features": log_summaries,
        "joined_features": joined,
        "total_seconds": round(time.perf_counter() - started, 2),
        "leakage_guard": "Drain and direct-template frequencies fitted on train.parquet only",
    }
    (args.output_dir / "v4_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
