from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.content_features import (  # noqa: E402
    DEFAULT_HASH_BUCKETS,
)
from soc_threat.content_model import ContentThreatModel, ModelMode  # noqa: E402
from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from train_npu_tabular import (  # noqa: E402
    choose_device,
    class_weights,
    fit_preprocessor,
    transform,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_indices(
    labels: np.ndarray, max_rows: int | None, seed: int
) -> np.ndarray:
    if max_rows is None or max_rows >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    counts = {label: int(np.sum(labels == label)) for label in LABELS}
    selected: list[np.ndarray] = []
    remaining = max_rows
    for index, label in enumerate(LABELS):
        candidates = np.flatnonzero(labels == label)
        if index == len(LABELS) - 1:
            take = min(len(candidates), remaining)
        else:
            target = round(max_rows * counts[label] / len(labels))
            take = min(len(candidates), max(1, target), remaining)
        selected.append(rng.choice(candidates, size=take, replace=False))
        remaining -= take
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return indices.astype(np.int64, copy=False)


def fixed_list_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.int32).reshape(-1, array.type.list_size)
    rows = array.to_pylist()
    if not rows:
        return np.empty((0, 0), dtype=np.int32)
    sizes = {len(value) for value in rows}
    if len(sizes) != 1:
        raise ValueError("Content token lists must all have the same fixed length")
    return np.asarray(rows, dtype=np.int32)


def read_content_data(
    path: Path,
    *,
    token_column: str,
    use_structured: bool,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["event_id", "label_binary", token_column]
    if use_structured:
        columns.extend(CATEGORICAL_FEATURES)
        columns.extend(NUMERIC_FEATURES)
    table = pq.read_table(path, columns=columns)
    label_values = np.asarray(table["label_binary"].to_pylist(), dtype=object)
    indices = stratified_indices(label_values, max_rows, seed)
    if len(indices) != len(table):
        table = table.take(pa.array(indices))
    tokens = fixed_list_to_numpy(table[token_column])
    frame_columns = ["event_id", "label_binary"]
    if use_structured:
        frame_columns.extend(CATEGORICAL_FEATURES)
        frame_columns.extend(NUMERIC_FEATURES)
    frame = table.select(frame_columns).to_pandas()
    return frame, tokens


def make_loader(
    categorical: np.ndarray,
    numeric: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.array(categorical, copy=True)),
        torch.from_numpy(np.array(numeric, copy=True)),
        torch.from_numpy(np.array(tokens, copy=True)),
        torch.from_numpy(np.array(labels, copy=True)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=shuffle and len(dataset) >= batch_size,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_probabilities(
    model: ContentThreatModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    fused_parts: list[np.ndarray] = []
    content_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for categorical, numeric, tokens, labels in loader:
        categorical = categorical.to(device, dtype=torch.long, non_blocking=True)
        numeric = numeric.to(device, dtype=torch.float32, non_blocking=True)
        tokens = tokens.to(device, dtype=torch.long, non_blocking=True)
        fused_logits, content_logits = model(categorical, numeric, tokens)
        fused_parts.append(torch.softmax(fused_logits, dim=1).cpu().numpy())
        content_parts.append(torch.softmax(content_logits, dim=1).cpu().numpy())
        label_parts.append(labels.numpy())
    return (
        np.concatenate(fused_parts),
        np.concatenate(content_parts),
        np.concatenate(label_parts),
    )


def empty_structured(rows: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((rows, 0), dtype=np.int64),
        np.empty((rows, 0), dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V6 content-only or structure/content fusion neural model"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("content", "fusion_raw", "fusion_field"),
        required=True,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--valid-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    parser.add_argument("--content-embedding-dim", type=int, default=64)
    parser.add_argument("--content-output-dim", type=int, default=128)
    parser.add_argument("--content-aux-weight", type=float, default=0.25)
    parser.add_argument("--token-dropout", type=float, default=0.05)
    parser.add_argument("--category-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train, args.valid):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 2 or args.valid_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    mode: ModelMode = args.mode
    use_structured = mode != "content"
    token_column = "field_token_ids" if mode == "fusion_field" else "raw_token_ids"

    data_started = time.perf_counter()
    train_frame, train_tokens = read_content_data(
        args.train,
        token_column=token_column,
        use_structured=use_structured,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    valid_frame, valid_tokens = read_content_data(
        args.valid,
        token_column=token_column,
        use_structured=use_structured,
        max_rows=args.max_valid_rows,
        seed=args.seed + 100,
    )
    if int(train_tokens.max(initial=0)) >= args.hash_buckets:
        raise ValueError("Prepared token ID exceeds --hash-buckets")

    if use_structured:
        preprocessor = fit_preprocessor(
            train_frame,
            list(CATEGORICAL_FEATURES),
            list(NUMERIC_FEATURES),
        )
        train_cat, train_num, train_y = transform(train_frame, preprocessor)
        valid_cat, valid_num, valid_y = transform(valid_frame, preprocessor)
        cardinalities = list(preprocessor["cardinalities"])
        numeric_count = len(NUMERIC_FEATURES)
    else:
        preprocessor = {
            "categorical_features": [],
            "numeric_features": [],
            "category_maps": {},
            "cardinalities": [],
            "numeric_means": {},
            "numeric_stds": {},
            "labels": LABELS,
        }
        train_cat, train_num = empty_structured(len(train_frame))
        valid_cat, valid_num = empty_structured(len(valid_frame))
        label_to_index = {label: index for index, label in enumerate(LABELS)}
        train_y = (
            train_frame["label_binary"].map(label_to_index).to_numpy(dtype=np.int64)
        )
        valid_y = (
            valid_frame["label_binary"].map(label_to_index).to_numpy(dtype=np.int64)
        )
        cardinalities = []
        numeric_count = 0
    data_seconds = time.perf_counter() - data_started

    train_loader = make_loader(
        train_cat,
        train_num,
        train_tokens,
        train_y,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_cat,
        valid_num,
        valid_tokens,
        valid_y,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ContentThreatModel(
        mode=mode,
        hash_buckets=args.hash_buckets,
        content_embedding_dim=args.content_embedding_dim,
        content_output_dim=args.content_output_dim,
        cardinalities=cardinalities,
        numeric_count=numeric_count,
        class_count=len(LABELS),
        token_dropout=args.token_dropout,
        category_dropout=args.category_dropout,
    ).to(device)
    weights = class_weights(train_y, args.class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_score = -1.0
    best_content_score = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    train_started = time.perf_counter()
    print(
        json.dumps(
            {
                "device": str(device),
                "mode": mode,
                "token_column": token_column,
                "train_rows": len(train_frame),
                "valid_rows": len(valid_frame),
                "hash_buckets": args.hash_buckets,
                "token_width": int(train_tokens.shape[1]),
                "class_weights": dict(zip(LABELS, weights.astype(float).tolist())),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        row_count = 0
        for categorical, numeric, tokens, labels in train_loader:
            categorical = categorical.to(device, dtype=torch.long, non_blocking=True)
            numeric = numeric.to(device, dtype=torch.float32, non_blocking=True)
            tokens = tokens.to(device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            fused_logits, content_logits = model(categorical, numeric, tokens)
            loss = criterion(fused_logits, labels)
            if mode != "content" and args.content_aux_weight > 0:
                loss = loss + args.content_aux_weight * criterion(
                    content_logits, labels
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_rows = labels.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_rows
            row_count += batch_rows

        fused_probabilities, content_probabilities, actual_indices = (
            predict_probabilities(model, valid_loader, device)
        )
        actual_labels = np.asarray(LABELS, dtype=object)[actual_indices]
        fused_labels = np.asarray(LABELS, dtype=object)[
            fused_probabilities.argmax(axis=1)
        ]
        content_labels = np.asarray(LABELS, dtype=object)[
            content_probabilities.argmax(axis=1)
        ]
        fused_metrics = evaluate_predictions(
            actual_labels,
            fused_labels,
            labels=LABELS,
            probabilities=fused_probabilities,
        )
        content_metrics = evaluate_predictions(
            actual_labels,
            content_labels,
            labels=LABELS,
            probabilities=content_probabilities,
        )
        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, row_count),
            "competition_score": fused_metrics["competition_score"],
            "macro_f1": fused_metrics["macro_f1"],
            "malicious_recall": fused_metrics["per_class"]["malicious"]["recall"],
            "content_competition_score": content_metrics["competition_score"],
            "content_macro_f1": content_metrics["macro_f1"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        score = float(fused_metrics["competition_score"])
        content_score = float(content_metrics["competition_score"])
        is_better = score > best_score + 1.0e-8 or (
            abs(score - best_score) <= 1.0e-8
            and content_score > best_content_score + 1.0e-8
        )
        if is_better:
            best_score = score
            best_content_score = content_score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    model.to(device)
    fused_probabilities, content_probabilities, actual_indices = predict_probabilities(
        model, valid_loader, device
    )
    actual_labels = np.asarray(LABELS, dtype=object)[actual_indices]
    predicted_labels = np.asarray(LABELS, dtype=object)[
        fused_probabilities.argmax(axis=1)
    ]
    content_labels = np.asarray(LABELS, dtype=object)[
        content_probabilities.argmax(axis=1)
    ]
    metrics = evaluate_predictions(
        actual_labels,
        predicted_labels,
        labels=LABELS,
        probabilities=fused_probabilities,
    )
    content_metrics = evaluate_predictions(
        actual_labels,
        content_labels,
        labels=LABELS,
        probabilities=content_probabilities,
    )
    metrics.update(
        {
            "model": "v6_content_aware_neural",
            "mode": mode,
            "device": str(device),
            "best_epoch": best_epoch,
            "selection_metric": "competition_score",
            "best_selection_score": best_score,
            "best_content_head_score": best_content_score,
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "data_seconds": data_seconds,
            "train_seconds": time.perf_counter() - train_started,
            "content_head_metrics": content_metrics,
            "class_weights": dict(zip(LABELS, weights.astype(float).tolist())),
            "history": history,
            "arguments": vars(args)
            | {
                "train": str(args.train),
                "valid": str(args.valid),
                "output_dir": str(args.output_dir),
            },
        }
    )

    checkpoint = {
        "model_state": best_state,
        "model_config": {
            "mode": mode,
            "hash_buckets": args.hash_buckets,
            "content_embedding_dim": args.content_embedding_dim,
            "content_output_dim": args.content_output_dim,
            "cardinalities": cardinalities,
            "numeric_count": numeric_count,
            "class_count": len(LABELS),
            "token_dropout": 0.0,
            "category_dropout": 0.0,
        },
        "preprocessor": preprocessor,
        "token_column": token_column,
        "labels": LABELS,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_table = pa.table(
        {
            "event_id": valid_frame["event_id"].astype(str).tolist(),
            "true_label": actual_labels.tolist(),
            "pred_label": predicted_labels.tolist(),
            "prob_benign": fused_probabilities[:, 0],
            "prob_malicious": fused_probabilities[:, 1],
            "prob_suspicious": fused_probabilities[:, 2],
            "content_pred_label": content_labels.tolist(),
            "content_prob_benign": content_probabilities[:, 0],
            "content_prob_malicious": content_probabilities[:, 1],
            "content_prob_suspicious": content_probabilities[:, 2],
        }
    )
    pq.write_table(
        prediction_table,
        args.output_dir / "valid_predictions.parquet",
        compression="zstd",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
