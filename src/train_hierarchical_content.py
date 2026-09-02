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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.content_features import DEFAULT_HASH_BUCKETS  # noqa: E402
from soc_threat.hierarchical_features import (  # noqa: E402
    HIERARCHICAL_REQUIRED_COLUMNS,
    HierarchicalArrays,
    fit_hierarchical_preprocessor,
    transform_hierarchical_inputs,
)
from soc_threat.hierarchical_model import (  # noqa: E402
    ContentInputMode,
    HierarchicalContentModel,
    NoveltyGateMode,
)
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from train_content_neural import fixed_list_to_numpy, stratified_indices  # noqa: E402
from train_npu_tabular import choose_device  # noqa: E402


THREAT_LABELS = ["benign", "threat"]
SUBTYPE_LABELS = ["malicious", "suspicious"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_data(
    path: Path,
    *,
    token_column: str,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = [
        "event_id",
        "label_binary",
        token_column,
        *HIERARCHICAL_REQUIRED_COLUMNS,
    ]
    columns = list(dict.fromkeys(columns))
    table = pq.read_table(path, columns=columns)
    labels = np.asarray(table["label_binary"].to_pylist(), dtype=object)
    indices = stratified_indices(labels, max_rows, seed)
    if len(indices) != len(table):
        table = table.take(pa.array(indices))
    tokens = fixed_list_to_numpy(table[token_column])
    frame = table.drop([token_column]).to_pandas()
    return frame, tokens


def class_indices(frame: pd.DataFrame) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    encoded = frame["label_binary"].map(label_to_index)
    if encoded.isna().any():
        unknown = sorted(frame.loc[encoded.isna(), "label_binary"].unique().tolist())
        raise ValueError(f"Unknown labels: {unknown}")
    return encoded.to_numpy(dtype=np.int64, copy=True)


def binary_class_weights(labels: np.ndarray, power: float) -> np.ndarray:
    if power < 0:
        raise ValueError("class weight power must be non-negative")
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(
            f"Both binary classes are required, got counts={counts.tolist()}"
        )
    balanced = len(labels) / (2.0 * counts)
    return np.power(balanced, power).astype(np.float32)


def make_loader(
    arrays: HierarchicalArrays,
    tokens: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.array(arrays.metadata_categorical, copy=True)),
        torch.from_numpy(np.array(arrays.metadata_numeric, copy=True)),
        torch.from_numpy(np.array(arrays.semantic_categorical, copy=True)),
        torch.from_numpy(np.array(arrays.semantic_numeric, copy=True)),
        torch.from_numpy(np.array(tokens, copy=True)),
        torch.from_numpy(np.array(arrays.novelty_gate, copy=True)),
        torch.from_numpy(np.array(arrays.combo_counts, copy=True)),
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


def move_inputs(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    (
        metadata_categorical,
        metadata_numeric,
        semantic_categorical,
        semantic_numeric,
        tokens,
        novelty_gate,
        combo_counts,
        labels,
    ) = batch
    return (
        metadata_categorical.to(device, dtype=torch.long, non_blocking=True),
        metadata_numeric.to(device, dtype=torch.float32, non_blocking=True),
        semantic_categorical.to(device, dtype=torch.long, non_blocking=True),
        semantic_numeric.to(device, dtype=torch.float32, non_blocking=True),
        tokens.to(device, dtype=torch.long, non_blocking=True),
        novelty_gate.to(device, dtype=torch.float32, non_blocking=True),
        combo_counts,
        labels.to(device, dtype=torch.long, non_blocking=True),
    )


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    float_mask = mask.to(losses.dtype)
    return (losses * float_mask).sum() / float_mask.sum().clamp_min(1.0)


def positive_evidence_preservation_loss(
    final_logits: torch.Tensor,
    metadata_logits: torch.Tensor,
    content_logits: torch.Tensor,
    threat_targets: torch.Tensor,
    *,
    positive_margin: float,
    allowed_branch_gap: float,
) -> torch.Tensor:
    """Keep the fused positive logit from erasing a useful branch on threats.

    Auxiliary branch logits are detached so this term cannot obtain a low loss by
    weakening the branch evidence that it is intended to preserve.
    """

    positive_mask = threat_targets == 1
    final_margin = final_logits[:, 1] - final_logits[:, 0]
    metadata_margin = metadata_logits[:, 1] - metadata_logits[:, 0]
    content_margin = content_logits[:, 1] - content_logits[:, 0]
    strongest_branch = torch.maximum(metadata_margin, content_margin).detach()
    floor = torch.full_like(strongest_branch, float(positive_margin))
    target_margin = torch.maximum(
        floor, strongest_branch - float(allowed_branch_gap)
    )
    losses = F.relu(target_margin - final_margin)
    float_mask = positive_mask.to(losses.dtype)
    return (losses * float_mask).sum() / float_mask.sum().clamp_min(1.0)


def binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-15, precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def hierarchical_probabilities(
    threat_probabilities: np.ndarray,
    subtype_probabilities: np.ndarray,
) -> np.ndarray:
    combined = np.empty((len(threat_probabilities), 3), dtype=np.float32)
    combined[:, 0] = threat_probabilities[:, 0]
    combined[:, 1:] = threat_probabilities[:, 1, np.newaxis] * subtype_probabilities
    return combined


def hierarchical_predictions(
    threat_probabilities: np.ndarray,
    subtype_probabilities: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threat_predicted = (threat_probabilities[:, 1] >= threshold).astype(np.int64)
    subtype_predicted = subtype_probabilities.argmax(axis=1).astype(np.int64)
    class_predicted = np.where(threat_predicted == 0, 0, subtype_predicted + 1)
    return class_predicted, threat_predicted, subtype_predicted


@torch.no_grad()
def predict(
    model: HierarchicalContentModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        "threat": [],
        "subtype": [],
        "metadata_threat": [],
        "content_threat": [],
        "metadata_subtype": [],
        "gate": [],
        "combo_counts": [],
        "labels": [],
    }
    for batch in loader:
        moved = move_inputs(batch, device)
        (
            metadata_categorical,
            metadata_numeric,
            semantic_categorical,
            semantic_numeric,
            tokens,
            novelty_gate,
            combo_counts,
            labels,
        ) = moved
        output = model(
            metadata_categorical,
            metadata_numeric,
            semantic_categorical,
            semantic_numeric,
            tokens,
            novelty_gate,
        )
        parts["threat"].append(torch.softmax(output.threat_logits, dim=1).cpu().numpy())
        parts["subtype"].append(
            torch.softmax(output.subtype_logits, dim=1).cpu().numpy()
        )
        parts["metadata_threat"].append(
            torch.softmax(output.metadata_threat_logits, dim=1).cpu().numpy()
        )
        parts["content_threat"].append(
            torch.softmax(output.content_threat_logits, dim=1).cpu().numpy()
        )
        parts["metadata_subtype"].append(
            torch.softmax(output.metadata_subtype_logits, dim=1).cpu().numpy()
        )
        parts["gate"].append(output.applied_gate.cpu().numpy())
        parts["combo_counts"].append(combo_counts.numpy())
        parts["labels"].append(labels.cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def evaluate_outputs(
    outputs: dict[str, np.ndarray],
    *,
    threat_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = hierarchical_probabilities(outputs["threat"], outputs["subtype"])
    predicted, threat_predicted, subtype_predicted = hierarchical_predictions(
        outputs["threat"], outputs["subtype"], threshold=threat_threshold
    )
    actual = outputs["labels"].astype(np.int64, copy=False)
    actual_names = np.asarray(LABELS, dtype=object)[actual]
    predicted_names = np.asarray(LABELS, dtype=object)[predicted]
    metrics = evaluate_predictions(
        actual_names,
        predicted_names,
        labels=LABELS,
        probabilities=probabilities,
    )
    actual_threat = (actual != 0).astype(np.int64)
    metadata_threat_predicted = outputs["metadata_threat"].argmax(axis=1)
    content_threat_predicted = outputs["content_threat"].argmax(axis=1)
    metadata_subtype_predicted = outputs["metadata_subtype"].argmax(axis=1)
    threat_mask = actual_threat == 1
    metrics["hierarchical_audit"] = {
        "threat_threshold": float(threat_threshold),
        "threat": binary_metrics(actual_threat, threat_predicted),
        "metadata_threat": binary_metrics(actual_threat, metadata_threat_predicted),
        "content_threat": binary_metrics(actual_threat, content_threat_predicted),
        "subtype_accuracy_on_true_threat": float(
            np.mean(subtype_predicted[threat_mask] == actual[threat_mask] - 1)
        ),
        "metadata_subtype_accuracy_on_true_threat": float(
            np.mean(metadata_subtype_predicted[threat_mask] == actual[threat_mask] - 1)
        ),
        "unseen_combo_rows": int(np.sum(outputs["combo_counts"] == 0)),
        "unseen_combo_errors": int(
            np.sum((outputs["combo_counts"] == 0) & (actual != predicted))
        ),
    }
    predictions = {
        "actual": actual,
        "predicted": predicted,
        "threat_predicted": threat_predicted,
        "subtype_predicted": subtype_predicted,
        "metadata_subtype_predicted": metadata_subtype_predicted,
        "probabilities": probabilities,
    }
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hierarchical threat/content neural model"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--novelty-gate", choices=("none", "count"), required=True)
    parser.add_argument(
        "--content-input", choices=("raw", "multiview"), default="raw"
    )
    parser.add_argument("--token-column")
    parser.add_argument("--content-view-count", type=int, default=4)
    parser.add_argument("--content-tokens-per-view", type=int, default=64)
    parser.add_argument(
        "--evidence-preservation-weight", type=float, default=0.0
    )
    parser.add_argument("--positive-threat-margin", type=float, default=0.0)
    parser.add_argument("--allowed-branch-logit-gap", type=float, default=0.5)
    parser.add_argument("--experiment-version", choices=("v7", "v8"), default="v7")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--valid-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--threat-class-weight-power", type=float, default=0.25)
    parser.add_argument("--subtype-class-weight-power", type=float, default=0.0)
    parser.add_argument("--subtype-loss-weight", type=float, default=0.75)
    parser.add_argument("--metadata-threat-aux-weight", type=float, default=0.15)
    parser.add_argument("--content-threat-aux-weight", type=float, default=0.25)
    parser.add_argument("--metadata-subtype-aux-weight", type=float, default=0.35)
    parser.add_argument("--threat-threshold", type=float, default=0.5)
    parser.add_argument("--novelty-pseudocount", type=float, default=32.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    parser.add_argument("--content-embedding-dim", type=int, default=64)
    parser.add_argument("--content-output-dim", type=int, default=128)
    parser.add_argument("--token-dropout", type=float, default=0.05)
    parser.add_argument("--category-dropout", type=float, default=0.02)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train, args.valid):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 < args.threat_threshold < 1.0:
        raise ValueError("threat threshold must be between zero and one")
    if args.evidence_preservation_weight < 0 or args.allowed_branch_logit_gap < 0:
        raise ValueError("Evidence preservation weights and gaps must be non-negative")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    gate_mode: NoveltyGateMode = args.novelty_gate
    content_input_mode: ContentInputMode = args.content_input
    token_column = args.token_column or (
        "multiview_token_ids"
        if content_input_mode == "multiview"
        else "raw_token_ids"
    )

    data_started = time.perf_counter()
    train_frame, train_tokens = read_data(
        args.train,
        token_column=token_column,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    valid_frame, valid_tokens = read_data(
        args.valid,
        token_column=token_column,
        max_rows=args.max_valid_rows,
        seed=args.seed + 100,
    )
    if int(train_tokens.max(initial=0)) >= args.hash_buckets:
        raise ValueError("Prepared token ID exceeds --hash-buckets")
    if content_input_mode == "multiview":
        expected_width = args.content_view_count * args.content_tokens_per_view
        if (
            train_tokens.shape[1] != expected_width
            or valid_tokens.shape[1] != expected_width
        ):
            raise ValueError(
                f"Expected {expected_width} multi-view tokens, "
                f"got train={train_tokens.shape[1]}, valid={valid_tokens.shape[1]}"
            )
    preprocessor = fit_hierarchical_preprocessor(
        train_frame, novelty_pseudocount=args.novelty_pseudocount
    )
    train_arrays = transform_hierarchical_inputs(train_frame, preprocessor)
    valid_arrays = transform_hierarchical_inputs(valid_frame, preprocessor)
    train_y = class_indices(train_frame)
    valid_y = class_indices(valid_frame)
    train_loader = make_loader(
        train_arrays,
        train_tokens,
        train_y,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_arrays,
        valid_tokens,
        valid_y,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    data_seconds = time.perf_counter() - data_started

    model = HierarchicalContentModel(
        novelty_gate_mode=gate_mode,
        hash_buckets=args.hash_buckets,
        content_embedding_dim=args.content_embedding_dim,
        content_output_dim=args.content_output_dim,
        metadata_cardinalities=list(preprocessor["metadata"]["cardinalities"]),
        metadata_numeric_count=len(preprocessor["metadata"]["numeric_features"]),
        semantic_cardinalities=list(preprocessor["semantic"]["cardinalities"]),
        semantic_numeric_count=len(preprocessor["semantic"]["numeric_features"]),
        token_dropout=args.token_dropout,
        category_dropout=args.category_dropout,
        content_input_mode=content_input_mode,
        content_view_count=args.content_view_count,
        content_tokens_per_view=args.content_tokens_per_view,
    ).to(device)

    train_threat = (train_y != 0).astype(np.int64)
    train_subtype = train_y[train_y != 0] - 1
    threat_weights = binary_class_weights(train_threat, args.threat_class_weight_power)
    subtype_weights = binary_class_weights(
        train_subtype, args.subtype_class_weight_power
    )
    threat_weight_tensor = torch.tensor(threat_weights, device=device)
    subtype_weight_tensor = torch.tensor(subtype_weights, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "novelty_gate": gate_mode,
                "content_input": content_input_mode,
                "token_column": token_column,
                "evidence_preservation_weight": args.evidence_preservation_weight,
                "train_rows": len(train_frame),
                "valid_rows": len(valid_frame),
                "train_combo_count": len(preprocessor["combo_counts"]),
                "valid_unseen_combo_rows": int(np.sum(valid_arrays.combo_counts == 0)),
                "threat_class_weights": dict(
                    zip(THREAT_LABELS, threat_weights.astype(float).tolist())
                ),
                "subtype_class_weights": dict(
                    zip(SUBTYPE_LABELS, subtype_weights.astype(float).tolist())
                ),
                "data_seconds": round(data_seconds, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_score = -1.0
    best_content_threat_f1 = -1.0
    best_log_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sums = {
            "total": 0.0,
            "threat": 0.0,
            "subtype": 0.0,
            "metadata_threat": 0.0,
            "content_threat": 0.0,
            "metadata_subtype": 0.0,
            "evidence_preservation": 0.0,
        }
        trained_rows = 0
        for batch in train_loader:
            moved = move_inputs(batch, device)
            (
                metadata_categorical,
                metadata_numeric,
                semantic_categorical,
                semantic_numeric,
                tokens,
                novelty_gate,
                _combo_counts,
                labels,
            ) = moved
            threat_targets = (labels != 0).to(torch.long)
            subtype_targets = (labels - 1).clamp_min(0)
            threat_mask = labels != 0
            optimizer.zero_grad(set_to_none=True)
            output = model(
                metadata_categorical,
                metadata_numeric,
                semantic_categorical,
                semantic_numeric,
                tokens,
                novelty_gate,
            )
            threat_loss = F.cross_entropy(
                output.threat_logits, threat_targets, weight=threat_weight_tensor
            )
            subtype_loss = masked_cross_entropy(
                output.subtype_logits,
                subtype_targets,
                threat_mask,
                subtype_weight_tensor,
            )
            metadata_threat_loss = F.cross_entropy(
                output.metadata_threat_logits,
                threat_targets,
                weight=threat_weight_tensor,
            )
            content_threat_loss = F.cross_entropy(
                output.content_threat_logits,
                threat_targets,
                weight=threat_weight_tensor,
            )
            metadata_subtype_loss = masked_cross_entropy(
                output.metadata_subtype_logits,
                subtype_targets,
                threat_mask,
                subtype_weight_tensor,
            )
            evidence_preservation_loss = positive_evidence_preservation_loss(
                output.threat_logits,
                output.metadata_threat_logits,
                output.content_threat_logits,
                threat_targets,
                positive_margin=args.positive_threat_margin,
                allowed_branch_gap=args.allowed_branch_logit_gap,
            )
            loss = (
                threat_loss
                + args.subtype_loss_weight * subtype_loss
                + args.metadata_threat_aux_weight * metadata_threat_loss
                + args.content_threat_aux_weight * content_threat_loss
                + args.metadata_subtype_aux_weight * metadata_subtype_loss
                + args.evidence_preservation_weight * evidence_preservation_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            rows = labels.shape[0]
            trained_rows += rows
            for name, value in (
                ("total", loss),
                ("threat", threat_loss),
                ("subtype", subtype_loss),
                ("metadata_threat", metadata_threat_loss),
                ("content_threat", content_threat_loss),
                ("metadata_subtype", metadata_subtype_loss),
                ("evidence_preservation", evidence_preservation_loss),
            ):
                loss_sums[name] += float(value.detach().cpu()) * rows

        outputs = predict(model, valid_loader, device)
        metrics, _ = evaluate_outputs(outputs, threat_threshold=args.threat_threshold)
        audit = metrics["hierarchical_audit"]
        epoch_result: dict[str, Any] = {
            "epoch": epoch,
            "competition_score": metrics["competition_score"],
            "macro_f1": metrics["macro_f1"],
            "malicious_recall": metrics["per_class"]["malicious"]["recall"],
            "threat_f1": audit["threat"]["f1"],
            "threat_false_positive": audit["threat"]["false_positive"],
            "threat_false_negative": audit["threat"]["false_negative"],
            "content_threat_f1": audit["content_threat"]["f1"],
            "subtype_accuracy": audit["subtype_accuracy_on_true_threat"],
            "unseen_combo_errors": audit["unseen_combo_errors"],
            "seconds": time.perf_counter() - epoch_started,
            "loss": {
                name: value / max(1, trained_rows) for name, value in loss_sums.items()
            },
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        score = float(metrics["competition_score"])
        content_threat_f1 = float(audit["content_threat"]["f1"])
        log_loss = float(metrics["multiclass_log_loss"])
        is_better = (
            score > best_score + 1.0e-8
            or (
                abs(score - best_score) <= 1.0e-8
                and content_threat_f1 > best_content_threat_f1 + 1.0e-8
            )
            or (
                abs(score - best_score) <= 1.0e-8
                and abs(content_threat_f1 - best_content_threat_f1) <= 1.0e-8
                and log_loss < best_log_loss - 1.0e-8
            )
        )
        if is_better:
            best_score = score
            best_content_threat_f1 = content_threat_f1
            best_log_loss = log_loss
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
    outputs = predict(model, valid_loader, device)
    metrics, predictions = evaluate_outputs(
        outputs, threat_threshold=args.threat_threshold
    )
    metrics.update(
        {
            "model": f"{args.experiment_version}_hierarchical_content_neural",
            "mode": (
                f"hierarchical_{content_input_mode}_{gate_mode}_"
                f"evidence_{args.evidence_preservation_weight:g}"
            ),
            "device": str(device),
            "best_epoch": best_epoch,
            "selection_metric": "competition_score",
            "best_selection_score": best_score,
            "best_content_threat_f1": best_content_threat_f1,
            "best_log_loss": best_log_loss,
            "train_rows": int(len(train_frame)),
            "valid_rows": int(len(valid_frame)),
            "data_seconds": data_seconds,
            "train_seconds": time.perf_counter() - train_started,
            "threat_class_weights": dict(
                zip(THREAT_LABELS, threat_weights.astype(float).tolist())
            ),
            "subtype_class_weights": dict(
                zip(SUBTYPE_LABELS, subtype_weights.astype(float).tolist())
            ),
            "history": history,
            "arguments": vars(args)
            | {
                "train": str(args.train),
                "valid": str(args.valid),
                "output_dir": str(args.output_dir),
                "resolved_token_column": token_column,
            },
        }
    )

    checkpoint = {
        "model_state": best_state,
        "model_config": {
            "novelty_gate_mode": gate_mode,
            "hash_buckets": args.hash_buckets,
            "content_embedding_dim": args.content_embedding_dim,
            "content_output_dim": args.content_output_dim,
            "metadata_cardinalities": list(preprocessor["metadata"]["cardinalities"]),
            "metadata_numeric_count": len(preprocessor["metadata"]["numeric_features"]),
            "semantic_cardinalities": list(preprocessor["semantic"]["cardinalities"]),
            "semantic_numeric_count": len(preprocessor["semantic"]["numeric_features"]),
            "token_dropout": 0.0,
            "category_dropout": 0.0,
            "content_input_mode": content_input_mode,
            "content_view_count": args.content_view_count,
            "content_tokens_per_view": args.content_tokens_per_view,
        },
        "threat_threshold": args.threat_threshold,
        "preprocessor": preprocessor,
        "labels": LABELS,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "preprocessor.json").write_text(
        json.dumps(preprocessor, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    actual_names = np.asarray(LABELS, dtype=object)[predictions["actual"]]
    predicted_names = np.asarray(LABELS, dtype=object)[predictions["predicted"]]
    subtype_names = np.asarray(SUBTYPE_LABELS, dtype=object)[
        predictions["subtype_predicted"]
    ]
    metadata_subtype_names = np.asarray(SUBTYPE_LABELS, dtype=object)[
        predictions["metadata_subtype_predicted"]
    ]
    probability = predictions["probabilities"]
    prediction_frame = pd.DataFrame(
        {
            "event_id": valid_frame["event_id"].astype(str).to_numpy(),
            "true_label": actual_names,
            "pred_label": predicted_names,
            "prob_benign": probability[:, 0],
            "prob_malicious": probability[:, 1],
            "prob_suspicious": probability[:, 2],
            "threat_probability": outputs["threat"][:, 1],
            "threat_predicted": predictions["threat_predicted"],
            "subtype_pred_label": subtype_names,
            "metadata_subtype_pred_label": metadata_subtype_names,
            "content_threat_probability": outputs["content_threat"][:, 1],
            "metadata_threat_probability": outputs["metadata_threat"][:, 1],
            "semantic_combo_count": outputs["combo_counts"].astype(np.int64),
            "novelty_gate": outputs["gate"],
        }
    )
    prediction_frame.to_parquet(
        args.output_dir / "valid_predictions.parquet", index=False
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
