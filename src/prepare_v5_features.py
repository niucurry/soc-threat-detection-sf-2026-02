from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import duckdb

from prepare_v4_features import (
    iter_raw_rows,
    join_base_and_log_features,
    write_log_features,
)
from soc_threat.log_semantics import DrainSettings
from soc_threat.v5_structured_semantics import V5GroupedDrainModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("SOC_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))
)


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
    model = V5GroupedDrainModel(settings)
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
                        "stage": "fit_v5_templates",
                        "rows": processed,
                        "groups": len(model.miners),
                        "schemas": len(model.schema_counts),
                        "semantic_templates": len(model.semantic_template_counts),
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
    print(
        json.dumps({"stage": "v5_template_model_ready", **summary}, ensure_ascii=False),
        flush=True,
    )
    return summary


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def audit_v5_features(path: Path) -> dict[str, Any]:
    connection = duckdb.connect()
    audit = connection.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            SUM(CASE WHEN payload_parse_status = 'success' THEN 1 ELSE 0 END)
                AS payload_parse_success,
            SUM(CASE WHEN payload_parse_status = 'failed' THEN 1 ELSE 0 END)
                AS payload_parse_failed,
            SUM(CASE WHEN schema_id <> '__MISSING__' THEN 1 ELSE 0 END)
                AS rows_with_schema,
            SUM(CASE WHEN schema_id <> '__MISSING__' AND schema_seen_train = 0
                     THEN 1 ELSE 0 END) AS unseen_schemas,
            SUM(CASE WHEN semantic_template_seen_train = 0 THEN 1 ELSE 0 END)
                AS unseen_semantic_templates,
            SUM(CASE WHEN malware_present = 1 THEN 1 ELSE 0 END)
                AS malware_rows,
            SUM(CASE WHEN authentication_present = 1 THEN 1 ELSE 0 END)
                AS authentication_rows
        FROM read_parquet('{_sql_path(path)}')
        """
    ).fetchone()
    format_rows = connection.execute(
        f"""
        SELECT message_format, payload_parse_status, COUNT(*) AS rows
        FROM read_parquet('{_sql_path(path)}')
        GROUP BY message_format, payload_parse_status
        ORDER BY message_format, payload_parse_status
        """
    ).fetchall()
    connection.close()
    return {
        "rows": int(audit[0]),
        "payload_parse_success": int(audit[1] or 0),
        "payload_parse_failed": int(audit[2] or 0),
        "rows_with_schema": int(audit[3] or 0),
        "unseen_schemas": int(audit[4] or 0),
        "unseen_semantic_templates": int(audit[5] or 0),
        "malware_rows": int(audit[6] or 0),
        "authentication_rows": int(audit[7] or 0),
        "format_status_rows": [
            {
                "message_format": row[0],
                "payload_parse_status": row[1],
                "rows": int(row[2]),
            }
            for row in format_rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare V5.1 schema-aware structured and grouped-Drain features"
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
        default=PROJECT_ROOT / "data" / "processed" / "v5",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v5_structured_neural" / "template_model",
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
        help="Load --model-dir instead of fitting V5 schema and Drain state again",
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
        model = V5GroupedDrainModel.load(args.model_dir)
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
        model = V5GroupedDrainModel.load(args.model_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_log = args.output_dir / "v5_train_log_features.parquet"
    valid_log = args.output_dir / "v5_valid_log_features.parquet"
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
    train_joined = args.output_dir / "v5_train.parquet"
    valid_joined = args.output_dir / "v5_valid.parquet"
    joined = [
        join_base_and_log_features(
            base_train, train_log, train_joined, force=args.force
        ),
        join_base_and_log_features(
            base_valid, valid_log, valid_joined, force=args.force
        ),
    ]
    summary = {
        "version": "v5.1",
        "template_model": template_manifest,
        "log_features": log_summaries,
        "joined_features": joined,
        "feature_audit": {
            "train": audit_v5_features(train_joined),
            "valid": audit_v5_features(valid_joined),
        },
        "total_seconds": round(time.perf_counter() - started, 2),
        "leakage_guard": (
            "Drain, schema frequencies, semantic-template frequencies, category vocabularies, "
            "and numeric normalization are fitted on train.parquet only"
        ),
    }
    (args.output_dir / "v5_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
