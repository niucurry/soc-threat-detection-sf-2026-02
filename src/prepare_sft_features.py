from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROMPT_FIELDS,
    engineer_structured_features,
    parse_soc_prompt,
)


def allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def split_is_valid(prompt: str, valid_percent: float) -> bool:
    digest = hashlib.blake2b(
        prompt.encode("utf-8"),
        digest_size=8,
        person=b"soc-v1-split",
    ).digest()
    bucket = int.from_bytes(digest, "big") % 10_000
    return bucket < int(round(valid_percent * 100))


def output_schema() -> pa.Schema:
    fields = [pa.field("event_id", pa.string()), pa.field("label_binary", pa.string())]
    fields.extend(pa.field(name, pa.string()) for name in CATEGORICAL_FEATURES)
    fields.extend(pa.field(name, pa.int64()) for name in NUMERIC_FEATURES)
    return pa.schema(fields)


class BufferedParquetWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.columns: dict[str, list[Any]] = {name: [] for name in schema.names}
        self.writer = pq.ParquetWriter(path, schema=schema, compression="zstd")
        self.rows = 0

    def append(self, row: dict[str, Any]) -> None:
        for name in self.schema.names:
            self.columns[name].append(row[name])
        self.rows += 1
        if len(self.columns["event_id"]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.columns["event_id"]:
            return
        table = pa.Table.from_pydict(self.columns, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.batch_size)
        self.columns = {name: [] for name in self.schema.names}

    def close(self) -> None:
        self.flush()
        self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream SFT CSV, recover structured features, and create a group-safe split"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--valid-percent", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 1.0 <= args.valid_percent <= 50.0:
        raise ValueError("--valid-percent must be between 1 and 50")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "sft_v1_train.parquet"
    valid_path = args.output_dir / "sft_v1_valid.parquet"
    manifest_path = args.output_dir / "sft_v1_manifest.json"
    for path in (train_path, valid_path, manifest_path):
        if path.exists():
            if not args.force:
                raise FileExistsError(f"{path} exists; pass --force to replace it")
            path.unlink()

    allow_large_csv_fields()
    schema = output_schema()
    train_writer = BufferedParquetWriter(train_path, schema, args.batch_size)
    valid_writer = BufferedParquetWriter(valid_path, schema, args.batch_size)
    label_counts = {"train": Counter(), "valid": Counter()}
    raw_missing_counts: Counter[str] = Counter()
    parse_error_examples: list[dict[str, Any]] = []
    invalid_labels: Counter[str] = Counter()
    rows_read = 0
    parsed_rows = 0
    started = time.perf_counter()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["system", "prompt", "response"]:
            raise ValueError(
                "Expected columns ['system', 'prompt', 'response'], "
                f"got {reader.fieldnames}"
            )
        for record in reader:
            if args.max_rows > 0 and rows_read >= args.max_rows:
                break
            rows_read += 1
            prompt = record.get("prompt") or ""
            label = (record.get("response") or "").strip().lower()
            if label not in LABELS:
                invalid_labels[label] += 1
                continue
            try:
                raw = parse_soc_prompt(prompt)
            except Exception as exc:
                if len(parse_error_examples) < 10:
                    parse_error_examples.append(
                        {
                            "row": rows_read,
                            "error": str(exc),
                            "prompt_preview": prompt[:500],
                        }
                    )
                continue

            for field in [*PROMPT_FIELDS.values(), "message_sanitized"]:
                if not raw[field]:
                    raw_missing_counts[field] += 1
            features = engineer_structured_features(raw)
            row = {
                "event_id": f"SFT-{rows_read:010d}",
                "label_binary": label,
                **features,
            }
            split_name = "valid" if split_is_valid(prompt, args.valid_percent) else "train"
            writer = valid_writer if split_name == "valid" else train_writer
            writer.append(row)
            label_counts[split_name][label] += 1
            parsed_rows += 1
            if rows_read % 100_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rows={rows_read} parsed={parsed_rows} "
                    f"seconds={elapsed:.1f}",
                    flush=True,
                )

    train_writer.close()
    valid_writer.close()
    elapsed = time.perf_counter() - started
    manifest = {
        "input": str(args.input.resolve()),
        "input_size_gb": round(args.input.stat().st_size / 1024**3, 3),
        "rows_read": rows_read,
        "parsed_rows": parsed_rows,
        "parse_error_count": rows_read - parsed_rows - sum(invalid_labels.values()),
        "parse_error_examples": parse_error_examples,
        "invalid_labels": dict(invalid_labels),
        "valid_percent_requested": args.valid_percent,
        "split_method": "blake2b(prompt) deterministic group split",
        "label_counts": {
            split: {
                label: int(counts.get(label, 0))
                for label in LABELS
            }
            for split, counts in label_counts.items()
        },
        "raw_missing_counts": dict(raw_missing_counts),
        "outputs": {
            "train": {
                "path": str(train_path.resolve()),
                "rows": train_writer.rows,
                "size_mb": round(train_path.stat().st_size / 1024**2, 2),
            },
            "valid": {
                "path": str(valid_path.resolve()),
                "rows": valid_writer.rows,
                "size_mb": round(valid_path.stat().st_size / 1024**2, 2),
            },
        },
        "elapsed_seconds": round(elapsed, 2),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if parse_error_examples or invalid_labels:
        raise RuntimeError(
            "Some rows could not be converted. Inspect sft_v1_manifest.json before training."
        )


if __name__ == "__main__":
    main()

