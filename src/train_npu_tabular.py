from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

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
from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as TABULAR_CATEGORICAL_FEATURES,
)
from soc_threat.feature_schema import NUMERIC_FEATURES as TABULAR_NUMERIC_FEATURES  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402


def resolve_feature_set(name: str) -> tuple[list[str], list[str]]:
    if name != "tabular":
        raise ValueError("This branch uses tabular features")
    return list(TABULAR_CATEGORICAL_FEATURES), list(TABULAR_NUMERIC_FEATURES)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def stratified_sample(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(frame):
        return frame
    proportions = frame["label_binary"].value_counts(normalize=True)
    parts: list[pd.DataFrame] = []
    remaining = max_rows
    labels = list(proportions.index)
    for index, label in enumerate(labels):
        group = frame.loc[frame["label_binary"] == label]
        if index == len(labels) - 1:
            take = remaining
        else:
            take = max(1, int(round(max_rows * float(proportions[label]))))
            take = min(take, len(group), remaining - (len(labels) - index - 1))
        parts.append(group.sample(n=take, random_state=seed + index))
        remaining -= take
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def read_frame(
    path: Path,
    max_rows: int | None,
    seed: int,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> pd.DataFrame:
    categorical_features = categorical_features or list(TABULAR_CATEGORICAL_FEATURES)
    numeric_features = numeric_features or list(TABULAR_NUMERIC_FEATURES)
    columns = ["event_id", "label_binary", *categorical_features, *numeric_features]
    frame = pd.read_parquet(path, columns=columns)
    return stratified_sample(frame, max_rows, seed)


def fit_preprocessor(
    train: pd.DataFrame,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> dict[str, Any]:
    categorical_features = categorical_features or list(TABULAR_CATEGORICAL_FEATURES)
    numeric_features = numeric_features or list(TABULAR_NUMERIC_FEATURES)
    category_maps: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for column in categorical_features:
        values = sorted(train[column].fillna("__MISSING__").astype(str).unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(values)}
        category_maps[column] = mapping
        # Index 0 is reserved for a category not observed during training.
        cardinalities.append(len(mapping) + 1)

    numeric = train[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
    means = numeric.mean(axis=0).astype(float)
    stds = numeric.std(axis=0).replace(0.0, 1.0).astype(float)
    return {
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "category_maps": category_maps,
        "cardinalities": cardinalities,
        "numeric_means": means.to_dict(),
        "numeric_stds": stds.to_dict(),
        "labels": LABELS,
    }


def transform(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical, numeric = transform_inputs(frame, preprocessor)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    labels = frame["label_binary"].map(label_to_index).to_numpy(dtype=np.int64, copy=True)
    return categorical, numeric, labels


def transform_inputs(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    categorical_columns: list[np.ndarray] = []
    categorical_features = list(preprocessor["categorical_features"])
    numeric_features = list(preprocessor["numeric_features"])
    for column in categorical_features:
        mapping = preprocessor["category_maps"][column]
        encoded = (
            frame[column]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )
        categorical_columns.append(encoded)
    categorical = np.column_stack(categorical_columns).astype(np.int64, copy=False)

    numeric_frame = frame[numeric_features].apply(pd.to_numeric, errors="coerce").fillna(-1.0)
    means = np.asarray(
        [preprocessor["numeric_means"][name] for name in numeric_features],
        dtype=np.float32,
    )
    stds = np.asarray(
        [preprocessor["numeric_stds"][name] for name in numeric_features],
        dtype=np.float32,
    )
    numeric = numeric_frame.to_numpy(dtype=np.float32, copy=True)
    numeric = np.clip((numeric - means) / stds, -12.0, 12.0).astype(np.float32)

    return categorical, numeric


def embedding_dimension(cardinality: int) -> int:
    return min(24, max(3, int(round(2.0 * cardinality**0.25))))


class TabularThreatModel(nn.Module):
    def __init__(self, cardinalities: list[int], numeric_count: int) -> None:
        super().__init__()
        dimensions = [embedding_dimension(value) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, dimensions)]
        )
        input_size = sum(dimensions) + numeric_count
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, len(LABELS)),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat([*embedded, numeric], dim=1))


def make_loader(
    categorical: np.ndarray,
    numeric: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(categorical),
        torch.from_numpy(numeric),
        torch.from_numpy(labels),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for categorical, numeric, labels in loader:
        categorical = categorical.to(device, non_blocking=True)
        numeric = numeric.to(device, non_blocking=True)
        logits = model(categorical, numeric)
        probability_parts.append(torch.softmax(logits, dim=1).cpu().numpy())
        label_parts.append(labels.numpy())
    return np.concatenate(probability_parts), np.concatenate(label_parts)


def class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    balanced = len(labels) / (len(LABELS) * counts)
    return np.power(balanced, power).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch-NPU tabular threat model")
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_0" / "tabular_train.parquet",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_0" / "tabular_valid.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v1_0_tabular",
    )
    parser.add_argument(
        "--feature-set",
        default="tabular",
        choices=("tabular",),
        help="tabular structural inputs",
    )
    parser.add_argument("--device", default="auto", help="auto, npu:0, cuda:0, or cpu")
    parser.add_argument("--selection-metric", choices=("competition_score", "macro_f1"), default="competition_score")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categorical_features, numeric_features = resolve_feature_set(args.feature_set)

    started = time.perf_counter()
    train_frame = read_frame(
        args.train,
        args.max_train_rows,
        args.seed,
        categorical_features,
        numeric_features,
    )
    valid_frame = read_frame(
        args.valid,
        args.max_valid_rows,
        args.seed + 100,
        categorical_features,
        numeric_features,
    )
    preprocessor = fit_preprocessor(
        train_frame,
        categorical_features,
        numeric_features,
    )
    train_cat, train_num, train_y = transform(train_frame, preprocessor)
    valid_cat, valid_num, valid_y = transform(valid_frame, preprocessor)
    data_seconds = time.perf_counter() - started

    train_loader = make_loader(
        train_cat,
        train_num,
        train_y,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_cat,
        valid_num,
        valid_y,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TabularThreatModel(preprocessor["cardinalities"], len(numeric_features)).to(device)
    weights = class_weights(train_y, args.class_weight_power)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    selection_metric = args.selection_metric
    best_selection_score = -1.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    train_started = time.perf_counter()

    print(
        json.dumps(
            {
                "device": str(device),
                "train_rows": len(train_frame),
                "valid_rows": len(valid_frame),
                "class_weights": dict(zip(LABELS, weights.astype(float).tolist())),
                "feature_set": args.feature_set,
                "cardinalities": dict(zip(categorical_features, preprocessor["cardinalities"])),
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
        for categorical, numeric, labels in train_loader:
            categorical = categorical.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical, numeric)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_rows = labels.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch_rows
            row_count += batch_rows

        probabilities, actual_indices = predict_probabilities(model, valid_loader, device)
        predicted_indices = probabilities.argmax(axis=1)
        actual_labels = np.asarray(LABELS, dtype=object)[actual_indices]
        predicted_labels = np.asarray(LABELS, dtype=object)[predicted_indices]
        metrics = evaluate_predictions(
            actual_labels,
            predicted_labels,
            labels=LABELS,
            probabilities=probabilities,
        )
        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, row_count),
            "macro_f1": metrics["macro_f1"],
            "competition_score": metrics["competition_score"],
            "accuracy": metrics["accuracy"],
            "benign_recall": metrics["per_class"]["benign"]["recall"],
            "malicious_recall": metrics["per_class"]["malicious"]["recall"],
            "suspicious_recall": metrics["per_class"]["suspicious"]["recall"],
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)

        if metrics[selection_metric] > best_selection_score + 1e-8:
            best_selection_score = metrics[selection_metric]
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
    probabilities, actual_indices = predict_probabilities(model, valid_loader, device)
    predicted_indices = probabilities.argmax(axis=1)
    actual_labels = np.asarray(LABELS, dtype=object)[actual_indices]
    predicted_labels = np.asarray(LABELS, dtype=object)[predicted_indices]
    metrics = evaluate_predictions(
        actual_labels,
        predicted_labels,
        labels=LABELS,
        probabilities=probabilities,
    )
    canonical_feature_set = args.feature_set
    metrics.update(
        {
            "model": {
                "tabular": "v1.0_tabular_embedding_mlp",
                "drain": "v1.1_tabular_mlp_with_grouped_drain",
                "structured": "v1.2_tabular_mlp_with_structured_parser",
            }[canonical_feature_set],
            "model_version": {
                "tabular": "v1.0",
                "drain": "v1.1",
                "structured": "v1.2",
            }[canonical_feature_set],
            "feature_set": canonical_feature_set,
            "device": str(device),
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "best_selection_score": best_selection_score,
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "data_seconds": data_seconds,
            "train_seconds": time.perf_counter() - train_started,
            "class_weights": dict(zip(LABELS, weights.astype(float).tolist())),
            "history": history,
            "arguments": vars(args) | {
                "train": str(args.train),
                "valid": str(args.valid),
                "output_dir": str(args.output_dir),
            },
        }
    )

    checkpoint = {
        "model_state": best_state,
        "cardinalities": preprocessor["cardinalities"],
        "numeric_count": len(numeric_features),
        "preprocessor": preprocessor,
        "labels": LABELS,
        "feature_set": args.feature_set,
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
            "prob_benign": probabilities[:, 0],
            "prob_malicious": probabilities[:, 1],
            "prob_suspicious": probabilities[:, 2],
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
