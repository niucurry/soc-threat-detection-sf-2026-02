from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_drain_features import join_base_and_log_features, write_log_features
from soc_threat.log_semantics import GroupedDrainModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich v1.0 features with a frozen v1.1 grouped-Drain model"
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw unlabeled parquet")
    parser.add_argument(
        "--base-features",
        type=Path,
        required=True,
        help="Output produced by prepare_inference_features.py",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-features", type=Path)
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.input, args.base_features, args.model_dir / "manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    model = GroupedDrainModel.load(args.model_dir)
    log_features = args.log_features or args.output.with_name(
        args.output.stem + "_log_features.parquet"
    )
    log_summary = write_log_features(
        args.input,
        log_features,
        model=model,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        force=args.force,
    )
    joined_summary = join_base_and_log_features(
        args.base_features,
        log_features,
        args.output,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "log_features": log_summary,
                "joined_features": joined_summary,
                "template_model": str(args.model_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
