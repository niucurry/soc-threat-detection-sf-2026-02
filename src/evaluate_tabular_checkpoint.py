from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from train_npu_tabular import (  # noqa: E402
    TabularThreatModel,
    choose_device,
    make_loader,
    predict_probabilities,
    read_frame,
    transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tabular model on official external validation data"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = choose_device(args.device)
    started = time.perf_counter()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    frame = read_frame(
        args.data,
        args.max_rows,
        seed=20260828,
        categorical_features=list(preprocessor["categorical_features"]),
        numeric_features=list(preprocessor["numeric_features"]),
    )
    categorical, numeric, labels = transform(frame, preprocessor)
    loader = make_loader(
        categorical,
        numeric,
        labels,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = TabularThreatModel(
        checkpoint["cardinalities"], checkpoint["numeric_count"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    probabilities, actual_indices = predict_probabilities(model, loader, device)
    predicted_indices = probabilities.argmax(axis=1)
    actual_labels = np.asarray(LABELS, dtype=object)[actual_indices]
    predicted_labels = np.asarray(LABELS, dtype=object)[predicted_indices]
    metrics = evaluate_predictions(
        actual_labels,
        predicted_labels,
        labels=LABELS,
        probabilities=probabilities,
    )
    metrics.update(
        {
            "evaluation_set": "official_external_validation",
            "model_path": str(args.model.resolve()),
            "data_path": str(args.data.resolve()),
            "device": str(device),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "official_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = pa.table(
        {
            "event_id": frame["event_id"].astype(str).tolist(),
            "true_label": actual_labels.tolist(),
            "pred_label": predicted_labels.tolist(),
            "prob_benign": probabilities[:, 0],
            "prob_malicious": probabilities[:, 1],
            "prob_suspicious": probabilities[:, 2],
        }
    )
    pq.write_table(
        predictions,
        args.output_dir / "official_predictions.parquet",
        compression="zstd",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
