from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from train_npu_tabular import (  # noqa: E402
    TabularThreatModel,
    choose_device,
    make_loader,
    predict_probabilities,
    transform_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an unlabeled feature parquet through a V1 checkpoint"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    columns = [
        "event_id",
        *preprocessor["categorical_features"],
        *preprocessor["numeric_features"],
    ]
    frame = pd.read_parquet(args.data, columns=columns)
    categorical, numeric = transform_inputs(frame, preprocessor)
    placeholder_labels = np.zeros(len(frame), dtype=np.int64)
    loader = make_loader(
        categorical,
        numeric,
        placeholder_labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = choose_device(args.device)
    model = TabularThreatModel(
        checkpoint["cardinalities"], checkpoint["numeric_count"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    probabilities, _ = predict_probabilities(model, loader, device)
    predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
    table = pa.table(
        {
            "event_id": frame["event_id"].astype(str).tolist(),
            "pred_label": predicted.tolist(),
            "prob_benign": probabilities[:, 0],
            "prob_malicious": probabilities[:, 1],
            "prob_suspicious": probabilities[:, 2],
        }
    )
    pq.write_table(table, args.output, compression="zstd")
    summary = {
        "rows": int(len(frame)),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "label_counts": {
            label: int(np.sum(predicted == label)) for label in LABELS
        },
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
