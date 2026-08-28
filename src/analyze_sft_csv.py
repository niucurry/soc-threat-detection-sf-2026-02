from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_LABELS = {"benign", "malicious", "suspicious"}


def allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def analyze(path: Path, max_rows: int) -> dict[str, Any]:
    allow_large_csv_fields()
    response_counts: Counter[str] = Counter()
    system_counts: Counter[str] = Counter()
    prompt_length_sum = 0
    prompt_length_min: int | None = None
    prompt_length_max = 0
    missing_value_rows = 0
    malformed_rows = 0
    first_record: dict[str, str] | None = None
    rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames != ["system", "prompt", "response"]:
            raise ValueError(
                f"Expected columns ['system', 'prompt', 'response'], got {fieldnames}"
            )

        for record in reader:
            if max_rows > 0 and rows >= max_rows:
                break
            rows += 1
            if None in record:
                malformed_rows += 1
                continue
            system = (record.get("system") or "").strip()
            prompt = record.get("prompt") or ""
            response = (record.get("response") or "").strip().lower()
            if not system or not prompt or not response:
                missing_value_rows += 1
            response_counts[response] += 1
            system_counts[system] += 1
            prompt_length = len(prompt)
            prompt_length_sum += prompt_length
            prompt_length_min = (
                prompt_length
                if prompt_length_min is None
                else min(prompt_length_min, prompt_length)
            )
            prompt_length_max = max(prompt_length_max, prompt_length)
            if first_record is None:
                first_record = {
                    "system_preview": system[:800],
                    "prompt_preview": prompt[:4000],
                    "response": response,
                    "prompt_length": str(prompt_length),
                }
            if rows % 100_000 == 0:
                print(f"scanned_rows={rows}", file=sys.stderr, flush=True)

    invalid_responses = {
        key: value
        for key, value in response_counts.items()
        if key not in EXPECTED_LABELS
    }
    label_counts = {
        label: int(response_counts.get(label, 0))
        for label in ("benign", "malicious", "suspicious")
    }
    return {
        "path": str(path.resolve()),
        "size_gb": round(path.stat().st_size / 1024**3, 3),
        "rows_scanned": rows,
        "scan_limit": max_rows if max_rows > 0 else "all",
        "label_counts": label_counts,
        "label_percent": {
            label: round(count * 100 / rows, 4) if rows else 0.0
            for label, count in label_counts.items()
        },
        "invalid_responses": invalid_responses,
        "unique_system_prompts_in_scan": len(system_counts),
        "missing_value_rows": missing_value_rows,
        "malformed_rows": malformed_rows,
        "prompt_length": {
            "minimum": prompt_length_min,
            "maximum": prompt_length_max,
            "average": round(prompt_length_sum / rows, 2) if rows else 0.0,
        },
        "first_record": first_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-analyze system/prompt/response CSV without loading it into memory"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Rows to scan; default 0 scans the full file for an unbiased label distribution",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path in addition to terminal output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.is_file():
        raise FileNotFoundError(args.path)
    result = analyze(args.path, args.max_rows)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
