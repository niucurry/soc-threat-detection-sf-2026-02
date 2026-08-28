from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAGIC_TYPES = {
    b"PAR1": "parquet",
    b"PK\x03\x04": "zip",
    b"\x1f\x8b": "gzip",
}


def detect_binary_type(prefix: bytes) -> str | None:
    for magic, file_type in MAGIC_TYPES.items():
        if prefix.startswith(magic):
            return file_type
    return None


def decode_sample(prefix: bytes) -> tuple[str | None, str | None]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return prefix.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def inspect(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(256 * 1024)

    binary_type = detect_binary_type(prefix)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": size,
        "size_mb": round(size / 1024**2, 2),
        "size_gb": round(size / 1024**3, 3),
        "detected_binary_type": binary_type,
    }

    if binary_type is not None:
        result["conclusion"] = (
            f"The file content is {binary_type}, regardless of its filename suffix."
        )
        if binary_type == "parquet" and path.suffix.lower() not in {".parquet", ".pq"}:
            result["recommended_action"] = "Rename the suffix to .parquet; do not parse it as CSV."
        elif binary_type in {"zip", "gzip"}:
            result["recommended_action"] = "Decompress the file before inspecting the contained data."
        return result

    text, encoding = decode_sample(prefix)
    result["text_encoding"] = encoding
    if text is None:
        result["conclusion"] = "The file is not recognized as ordinary CSV text or Parquet/ZIP/GZIP."
        result["first_32_bytes_hex"] = prefix[:32].hex()
        return result

    nonempty_lines = [line for line in text.splitlines() if line.strip()][:6]
    if not nonempty_lines:
        result["conclusion"] = "The sampled beginning of the file is empty."
        return result

    sample_text = "\n".join(nonempty_lines)
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t|;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = max(",\t|;", key=lambda value: nonempty_lines[0].count(value))

    rows = list(csv.reader(nonempty_lines, delimiter=delimiter))
    header = rows[0]
    result.update(
        {
            "detected_text_type": "delimited_text",
            "delimiter": {"\t": "TAB"}.get(delimiter, delimiter),
            "header_column_count": len(header),
            "header_columns": header,
            "sample_row_column_counts": [len(row) for row in rows[1:]],
            "sample_preview": [
                [value[:120] + ("..." if len(value) > 120 else "") for value in row]
                for row in rows[1:4]
            ],
        }
    )
    expected_columns = {
        "event_id",
        "timestamp",
        "pipeline",
        "src_ip",
        "dst_ip",
        "src_port",
        "src_host",
        "dst_host",
        "username",
        "message_sanitized",
        "product_name",
        "vendor_name",
    }
    header_set = set(header)
    result["matches_soc_log_schema"] = expected_columns.issubset(header_set)
    result["contains_label"] = "label_binary" in header_set
    split_candidates = [
        name for name in ("split", "dataset", "dataset_split", "source_file") if name in header_set
    ]
    result["split_indicator_columns"] = split_candidates

    if path.suffix.lower() == ".cvs":
        result["suffix_warning"] = ".cvs is usually a typo; standard CSV uses .csv."
    if result["matches_soc_log_schema"]:
        if split_candidates:
            result["conclusion"] = "This appears to be SOC CSV data with a train/valid split indicator."
        elif result["contains_label"]:
            result["conclusion"] = (
                "This appears to contain labeled SOC rows, but no explicit train/valid split column was found."
            )
        else:
            result["conclusion"] = "This appears to contain unlabeled SOC input rows."
    else:
        result["conclusion"] = "The header does not match the expected SOC log schema."
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a large CSV/CVS/Parquet-like file without loading it into memory"
    )
    parser.add_argument("path", type=Path, help="Path to the file on the cloud platform")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.is_file():
        raise FileNotFoundError(args.path)
    print(json.dumps(inspect(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

