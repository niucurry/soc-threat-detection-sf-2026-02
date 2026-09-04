from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.anchored_residual_model import (  # noqa: E402
    AnchoredConflictResidualModel,
    ResidualInputMode,
)
from soc_threat.hierarchical_features import (  # noqa: E402
    HIERARCHICAL_REQUIRED_COLUMNS,
    HierarchicalArrays,
    transform_hierarchical_inputs,
)
from soc_threat.hierarchical_model import HierarchicalContentModel  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from train_content_neural import fixed_list_to_numpy, stratified_indices  # noqa: E402
from train_hierarchical_content import (  # noqa: E402
    binary_metrics,
    hierarchical_predictions,
    hierarchical_probabilities,
)
from train_npu_tabular import choose_device  # noqa: E402


SUBTYPE_LABELS = ["malicious", "suspicious"]


@dataclass(frozen=True)
class ResidualData:
    frame: pd.DataFrame
    raw_tokens: np.ndarray
    multiview_tokens: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_data(
    path: Path,
    *,
    residual_input_mode: ResidualInputMode,
    max_rows: int | None,
    seed: int,
) -> ResidualData:
    columns = [
        "event_id",
        "label_binary",
        "raw_token_ids",
        *HIERARCHICAL_REQUIRED_COLUMNS,
    ]
    if residual_input_mode == "multiview":
        columns.append("multiview_token_ids")
    columns = list(dict.fromkeys(columns))
    table = pq.read_table(path, columns=columns)
    labels = np.asarray(table["label_binary"].to_pylist(), dtype=object)
    indices = stratified_indices(labels, max_rows, seed)
    if len(indices) != len(table):
        table = table.take(pa.array(indices))
    raw_tokens = fixed_list_to_numpy(table["raw_token_ids"])
    if residual_input_mode == "multiview":
        multiview_tokens = fixed_list_to_numpy(table["multiview_token_ids"])
        drop_columns = ["raw_token_ids", "multiview_token_ids"]
    else:
        multiview_tokens = np.zeros((len(table), 0), dtype=np.uint16)
        drop_columns = ["raw_token_ids"]
    frame = table.drop(drop_columns).to_pandas()
    return ResidualData(
        frame=frame,
        raw_tokens=raw_tokens,
        multiview_tokens=multiview_tokens,
    )


def class_indices(frame: pd.DataFrame) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(LABELS)}
    values = frame["label_binary"].map(mapping)
    if values.isna().any():
        unknown = sorted(frame.loc[values.isna(), "label_binary"].unique().tolist())
        raise ValueError(f"Unknown labels: {unknown}")
    return values.to_numpy(dtype=np.int64, copy=True)


def make_dataset(
    arrays: HierarchicalArrays,
    data: ResidualData,
    labels: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.array(arrays.metadata_categorical, copy=True)),
        torch.from_numpy(np.array(arrays.metadata_numeric, copy=True)),
        torch.from_numpy(np.array(arrays.semantic_categorical, copy=True)),
        torch.from_numpy(np.array(arrays.semantic_numeric, copy=True)),
        torch.from_numpy(np.array(data.raw_tokens, copy=True)),
        torch.from_numpy(np.array(data.multiview_tokens, copy=True)),
        torch.from_numpy(np.array(arrays.novelty_gate, copy=True)),
        torch.from_numpy(np.array(arrays.combo_counts, copy=True)),
        torch.from_numpy(np.array(labels, copy=True)),
    )


def make_loader(
    dataset: TensorDataset | Subset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    (
        metadata_categorical,
        metadata_numeric,
        semantic_categorical,
        semantic_numeric,
        raw_tokens,
        multiview_tokens,
        novelty_gate,
        combo_counts,
        labels,
    ) = batch
    return (
        metadata_categorical.to(device, dtype=torch.long, non_blocking=True),
        metadata_numeric.to(device, dtype=torch.float32, non_blocking=True),
        semantic_categorical.to(device, dtype=torch.long, non_blocking=True),
        semantic_numeric.to(device, dtype=torch.float32, non_blocking=True),
        raw_tokens.to(device, dtype=torch.long, non_blocking=True),
        multiview_tokens.to(device, dtype=torch.long, non_blocking=True),
        novelty_gate.to(device, dtype=torch.float32, non_blocking=True),
        combo_counts.to(device, dtype=torch.float32, non_blocking=True),
        labels.to(device, dtype=torch.long, non_blocking=True),
    )


def model_forward(
    model: AnchoredConflictResidualModel,
    moved: tuple[torch.Tensor, ...],
):
    (
        metadata_categorical,
        metadata_numeric,
        semantic_categorical,
        semantic_numeric,
        raw_tokens,
        multiview_tokens,
        novelty_gate,
        combo_counts,
        _labels,
    ) = moved
    optional_multiview = (
        multiview_tokens if model.residual_input_mode == "multiview" else None
    )
    return model(
        metadata_categorical,
        metadata_numeric,
        semantic_categorical,
        semantic_numeric,
        raw_tokens,
        optional_multiview,
        novelty_gate,
        combo_counts,
    )


@torch.no_grad()
def find_reliability_indices(
    model: AnchoredConflictResidualModel,
    loader: DataLoader,
    device: torch.device,
    *,
    hard_negative_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    margin_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for batch in loader:
        moved = move_batch(batch, device)
        frozen = model.anchor_forward(
            moved[0], moved[1], moved[2], moved[3], moved[4], moved[6]
        )
        metadata_margin = (
            frozen.metadata_threat_logits[:, 1]
            - frozen.metadata_threat_logits[:, 0]
        )
        margin_parts.append(metadata_margin.cpu().numpy())
        label_parts.append(moved[-1].cpu().numpy())
    margins = np.concatenate(margin_parts)
    labels = np.concatenate(label_parts)
    # Positive reliability examples are true threats for which the metadata
    # branch votes threat.  The in-sample auxiliary branch can have zero false
    # positives, so add the benign rows with the highest metadata margins as
    # hard negatives.  This teaches distrust without using validation labels.
    positive = np.flatnonzero((labels != 0) & (margins > 0)).astype(
        np.int64, copy=False
    )
    if len(positive) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    benign = np.flatnonzero(labels == 0).astype(np.int64, copy=False)
    negative_count = min(
        len(benign), int(np.ceil(len(positive) * hard_negative_ratio))
    )
    if negative_count:
        benign_margins = margins[benign]
        if negative_count == len(benign):
            hard_negative = benign
        else:
            positions = np.argpartition(benign_margins, -negative_count)[
                -negative_count:
            ]
            hard_negative = benign[positions]
        selected = np.concatenate([positive, hard_negative])
    else:
        selected = positive
    # A deterministic ordering is useful for reproducible Subset indexing;
    # DataLoader performs the requested epoch-level shuffle afterwards.
    selected.sort()
    return selected, labels[selected]


@torch.no_grad()
def predict(
    model: AnchoredConflictResidualModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    names = (
        "threat",
        "anchor_threat",
        "subtype",
        "metadata_threat",
        "content_threat",
        "metadata_subtype",
        "gate",
        "combo_counts",
        "metadata_candidate",
        "conflict",
        "trust_logit",
        "trust",
        "delta",
        "labels",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for batch in loader:
        moved = move_batch(batch, device)
        output = model_forward(model, moved)
        parts["threat"].append(
            torch.softmax(output.threat_logits, dim=1).cpu().numpy()
        )
        parts["anchor_threat"].append(
            torch.softmax(output.anchor_threat_logits, dim=1).cpu().numpy()
        )
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
        parts["combo_counts"].append(moved[7].cpu().numpy())
        parts["metadata_candidate"].append(
            output.metadata_candidate.cpu().numpy()
        )
        parts["conflict"].append(output.conflict_mask.cpu().numpy())
        parts["trust_logit"].append(output.trust_logit.cpu().numpy())
        parts["trust"].append(output.trust_score.cpu().numpy())
        parts["delta"].append(output.delta_margin.cpu().numpy())
        parts["labels"].append(moved[-1].cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def evaluate_outputs(
    outputs: dict[str, np.ndarray],
    *,
    threat_threshold: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    final_probabilities = hierarchical_probabilities(
        outputs["threat"], outputs["subtype"]
    )
    anchor_probabilities = hierarchical_probabilities(
        outputs["anchor_threat"], outputs["subtype"]
    )
    predicted, threat_predicted, subtype_predicted = hierarchical_predictions(
        outputs["threat"], outputs["subtype"], threshold=threat_threshold
    )
    anchor_predicted, anchor_threat_predicted, _ = hierarchical_predictions(
        outputs["anchor_threat"],
        outputs["subtype"],
        threshold=threat_threshold,
    )
    actual = outputs["labels"].astype(np.int64, copy=False)
    actual_names = np.asarray(LABELS, dtype=object)[actual]
    predicted_names = np.asarray(LABELS, dtype=object)[predicted]
    metrics = evaluate_predictions(
        actual_names,
        predicted_names,
        labels=LABELS,
        probabilities=final_probabilities,
    )
    actual_threat = (actual != 0).astype(np.int64)
    metadata_threat_predicted = outputs["metadata_threat"].argmax(axis=1)
    content_threat_predicted = outputs["content_threat"].argmax(axis=1)
    metadata_subtype_predicted = outputs["metadata_subtype"].argmax(axis=1)
    true_threat_mask = actual_threat == 1
    conflict = outputs["conflict"].astype(bool)
    metadata_candidate = outputs["metadata_candidate"].astype(bool)
    changed = predicted != anchor_predicted
    anchor_wrong = anchor_predicted != actual
    final_wrong = predicted != actual
    metrics["hierarchical_audit"] = {
        "threat_threshold": float(threat_threshold),
        "threat": binary_metrics(actual_threat, threat_predicted),
        "anchor_threat": binary_metrics(actual_threat, anchor_threat_predicted),
        "metadata_threat": binary_metrics(actual_threat, metadata_threat_predicted),
        "content_threat": binary_metrics(actual_threat, content_threat_predicted),
        "subtype_accuracy_on_true_threat": float(
            np.mean(subtype_predicted[true_threat_mask] == actual[true_threat_mask] - 1)
        ),
        "metadata_subtype_accuracy_on_true_threat": float(
            np.mean(
                metadata_subtype_predicted[true_threat_mask]
                == actual[true_threat_mask] - 1
            )
        ),
        "unseen_combo_rows": int(np.sum(outputs["combo_counts"] == 0)),
        "unseen_combo_errors": int(
            np.sum((outputs["combo_counts"] == 0) & final_wrong)
        ),
    }
    metrics["residual_audit"] = {
        "anchor_errors": int(np.sum(anchor_wrong)),
        "final_errors": int(np.sum(final_wrong)),
        "metadata_reliability_candidates": int(np.sum(metadata_candidate)),
        "metadata_candidate_true_threat": int(
            np.sum(metadata_candidate & (actual_threat == 1))
        ),
        "metadata_candidate_true_benign": int(
            np.sum(metadata_candidate & (actual_threat == 0))
        ),
        "conflict_candidates": int(np.sum(conflict)),
        "conflict_true_threat": int(np.sum(conflict & (actual_threat == 1))),
        "conflict_true_benign": int(np.sum(conflict & (actual_threat == 0))),
        "changed_predictions": int(np.sum(changed)),
        "fixed_anchor_errors": int(np.sum(anchor_wrong & ~final_wrong)),
        "new_errors": int(np.sum(~anchor_wrong & final_wrong)),
        "both_wrong": int(np.sum(anchor_wrong & final_wrong)),
        "positive_delta_rows": int(np.sum(outputs["delta"] > 0)),
        "negative_delta_rows": int(np.sum(outputs["delta"] < 0)),
        "mean_abs_delta_on_candidates": float(
            np.mean(np.abs(outputs["delta"][conflict])) if np.any(conflict) else 0.0
        ),
    }
    predictions = {
        "actual": actual,
        "predicted": predicted,
        "anchor_predicted": anchor_predicted,
        "threat_predicted": threat_predicted,
        "subtype_predicted": subtype_predicted,
        "metadata_subtype_predicted": metadata_subtype_predicted,
        "probabilities": final_probabilities,
        "anchor_probabilities": anchor_probabilities,
    }
    return metrics, predictions


def error_count(metrics: dict[str, Any]) -> int:
    matrix = metrics["confusion_matrix"]["counts"]
    return int(sum(map(sum, matrix)) - sum(matrix[i][i] for i in range(len(matrix))))


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_score = float(candidate["competition_score"])
    incumbent_score = float(incumbent["competition_score"])
    if candidate_score > incumbent_score + 1.0e-10:
        return True
    if abs(candidate_score - incumbent_score) > 1.0e-10:
        return False
    candidate_errors = error_count(candidate)
    incumbent_errors = error_count(incumbent)
    if candidate_errors != incumbent_errors:
        return candidate_errors < incumbent_errors
    return float(candidate["multiclass_log_loss"]) < float(
        incumbent["multiclass_log_loss"]
    ) - 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a zero-initialized conflict resolver anchored to V7"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--anchor-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-input", choices=("anchor", "multiview"), required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scan-batch-size", type=int, default=4096)
    parser.add_argument("--valid-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--trust-regularization-weight", type=float, default=0.01)
    parser.add_argument("--candidate-threat-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=2.0)
    parser.add_argument("--threat-threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--content-view-count", type=int, default=4)
    parser.add_argument("--content-tokens-per-view", type=int, default=64)
    parser.add_argument("--token-dropout", type=float, default=0.05)
    parser.add_argument("--residual-hidden-dim", type=int, default=128)
    parser.add_argument("--max-conflict-gap", type=float, default=24.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train, args.valid, args.anchor_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.epochs, args.batch_size, args.valid_batch_size, args.patience) < 1:
        raise ValueError("epoch, batch and patience settings must be positive")
    if args.distillation_weight < 0 or args.trust_regularization_weight < 0:
        raise ValueError("regularization weights must be non-negative")
    if args.candidate_threat_weight <= 0 or args.temperature <= 0:
        raise ValueError("candidate threat weight and temperature must be positive")
    if args.hard_negative_ratio < 0:
        raise ValueError("hard-negative-ratio must be non-negative")

    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.anchor_model, map_location="cpu", weights_only=False)
    anchor_config = dict(checkpoint["model_config"])
    anchor_config.setdefault("content_input_mode", "raw")
    anchor_config.setdefault("content_view_count", 4)
    anchor_config.setdefault("content_tokens_per_view", 64)
    anchor = HierarchicalContentModel(**anchor_config)
    anchor.load_state_dict(checkpoint["model_state"])

    residual_input_mode: ResidualInputMode = args.residual_input
    data_started = time.perf_counter()
    train_data = read_data(
        args.train,
        residual_input_mode=residual_input_mode,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    valid_data = read_data(
        args.valid,
        residual_input_mode=residual_input_mode,
        max_rows=args.max_valid_rows,
        seed=args.seed + 1,
    )
    preprocessor = checkpoint["preprocessor"]
    train_arrays = transform_hierarchical_inputs(train_data.frame, preprocessor)
    valid_arrays = transform_hierarchical_inputs(valid_data.frame, preprocessor)
    train_labels = class_indices(train_data.frame)
    valid_labels = class_indices(valid_data.frame)
    train_dataset = make_dataset(train_arrays, train_data, train_labels)
    valid_dataset = make_dataset(valid_arrays, valid_data, valid_labels)
    valid_event_ids = valid_data.frame["event_id"].astype(str).to_numpy(copy=True)
    # TensorDataset owns writable tensor copies.  Release Arrow/Pandas/NumPy
    # staging buffers before scanning the full data to limit peak cloud RAM.
    del train_data, valid_data, train_arrays, valid_arrays, train_labels, valid_labels
    gc.collect()
    scan_loader = make_loader(
        train_dataset,
        batch_size=args.scan_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AnchoredConflictResidualModel(
        anchor=anchor,
        residual_input_mode=residual_input_mode,
        hash_buckets=int(anchor_config["hash_buckets"]),
        content_embedding_dim=int(anchor_config["content_embedding_dim"]),
        content_output_dim=int(anchor_config["content_output_dim"]),
        token_dropout=args.token_dropout,
        content_view_count=args.content_view_count,
        content_tokens_per_view=args.content_tokens_per_view,
        residual_hidden_dim=args.residual_hidden_dim,
        max_conflict_gap=args.max_conflict_gap,
    ).to(device)
    candidate_indices, candidate_labels = find_reliability_indices(
        model,
        scan_loader,
        device,
        hard_negative_ratio=args.hard_negative_ratio,
    )
    if len(candidate_indices) == 0:
        raise ValueError("The frozen V7 metadata branch produced no threat candidates")
    candidate_dataset = Subset(train_dataset, candidate_indices.tolist())
    train_loader = make_loader(
        candidate_dataset,
        batch_size=min(args.batch_size, len(candidate_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
    )
    data_seconds = time.perf_counter() - data_started

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    class_weights = torch.tensor(
        [1.0, args.candidate_threat_weight], dtype=torch.float32, device=device
    )

    baseline_outputs = predict(model, valid_loader, device)
    best_metrics, _ = evaluate_outputs(
        baseline_outputs, threat_threshold=args.threat_threshold
    )
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "competition_score": best_metrics["competition_score"],
            "errors": error_count(best_metrics),
            "multiclass_log_loss": best_metrics["multiclass_log_loss"],
            "residual_audit": best_metrics["residual_audit"],
            "note": "exact frozen V7 anchor before residual training",
        }
    ]
    print(
        json.dumps(
            {
                "stage": "v9_anchor_ready",
                "device": str(device),
                "residual_input": residual_input_mode,
                "train_rows": len(train_dataset),
                "train_reliability_rows": len(candidate_indices),
                "train_hard_negative_benign": int(np.sum(candidate_labels == 0)),
                "train_metadata_positive_malicious": int(
                    np.sum(candidate_labels == 1)
                ),
                "train_metadata_positive_suspicious": int(
                    np.sum(candidate_labels == 2)
                ),
                "valid_rows": len(valid_dataset),
                "baseline_score": best_metrics["competition_score"],
                "baseline_errors": error_count(best_metrics),
                "data_seconds": round(data_seconds, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    stale_epochs = 0
    train_started = time.perf_counter()
    temperature = float(args.temperature)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sums = {
            "total": 0.0,
            "reliability": 0.0,
            "classification": 0.0,
            "distill": 0.0,
            "trust": 0.0,
        }
        trained_rows = 0
        for batch in train_loader:
            moved = move_batch(batch, device)
            labels = moved[-1]
            threat_targets = (labels != 0).to(torch.long)
            optimizer.zero_grad(set_to_none=True)
            output = model_forward(model, moved)
            reliability_loss = F.binary_cross_entropy_with_logits(
                output.trust_logit,
                threat_targets.to(output.trust_logit.dtype),
                pos_weight=class_weights[1],
            )
            classification_loss = F.cross_entropy(
                output.threat_logits, threat_targets, weight=class_weights
            )
            anchor_target = torch.softmax(
                output.anchor_threat_logits.detach() / temperature, dim=1
            )
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(output.threat_logits / temperature, dim=1),
                    anchor_target,
                    reduction="batchmean",
                )
                * temperature**2
            )
            trust_loss = torch.mean(output.trust_score.square())
            loss = (
                reliability_loss
                + classification_loss
                + args.distillation_weight * distillation_loss
                + args.trust_regularization_weight * trust_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
            optimizer.step()
            rows = labels.shape[0]
            trained_rows += rows
            for name, value in (
                ("total", loss),
                ("reliability", reliability_loss),
                ("classification", classification_loss),
                ("distill", distillation_loss),
                ("trust", trust_loss),
            ):
                loss_sums[name] += float(value.detach().cpu()) * rows

        outputs = predict(model, valid_loader, device)
        metrics, _ = evaluate_outputs(
            outputs, threat_threshold=args.threat_threshold
        )
        result = {
            "epoch": epoch,
            "competition_score": metrics["competition_score"],
            "errors": error_count(metrics),
            "multiclass_log_loss": metrics["multiclass_log_loss"],
            "malicious_recall": metrics["per_class"]["malicious"]["recall"],
            "threat_false_positive": metrics["hierarchical_audit"]["threat"]["false_positive"],
            "threat_false_negative": metrics["hierarchical_audit"]["threat"]["false_negative"],
            "residual_audit": metrics["residual_audit"],
            "loss": {
                name: value / max(1, trained_rows) for name, value in loss_sums.items()
            },
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if is_better(metrics, best_metrics):
            best_metrics = metrics
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

    model.load_state_dict(best_state)
    model.to(device)
    outputs = predict(model, valid_loader, device)
    metrics, predictions = evaluate_outputs(
        outputs, threat_threshold=args.threat_threshold
    )
    metrics.update(
        {
            "model": "v9_anchored_conflict_residual",
            "mode": f"anchored_conflict_{residual_input_mode}",
            "device": str(device),
            "best_epoch": best_epoch,
            "selection_metric": "competition_score_then_errors_then_log_loss",
            "epoch_zero_is_exact_v7_anchor": True,
            "train_rows": len(train_dataset),
            "train_reliability_rows": len(candidate_indices),
            "train_reliability_label_counts": {
                label: int(np.sum(candidate_labels == index))
                for index, label in enumerate(LABELS)
            },
            "valid_rows": len(valid_dataset),
            "data_seconds": data_seconds,
            "train_seconds": time.perf_counter() - train_started,
            "history": history,
            "arguments": vars(args)
            | {
                "train": str(args.train),
                "valid": str(args.valid),
                "anchor_model": str(args.anchor_model),
                "output_dir": str(args.output_dir),
            },
        }
    )

    saved_checkpoint = {
        "model_state": best_state,
        "model_config": {
            "anchor_config": anchor_config,
            "residual_input_mode": residual_input_mode,
            "content_view_count": args.content_view_count,
            "content_tokens_per_view": args.content_tokens_per_view,
            "residual_hidden_dim": args.residual_hidden_dim,
            "max_conflict_gap": args.max_conflict_gap,
            "token_dropout": 0.0,
        },
        "preprocessor": preprocessor,
        "threat_threshold": args.threat_threshold,
        "labels": LABELS,
        "source_anchor_model": str(args.anchor_model),
    }
    torch.save(saved_checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    actual_names = np.asarray(LABELS, dtype=object)[predictions["actual"]]
    predicted_names = np.asarray(LABELS, dtype=object)[predictions["predicted"]]
    anchor_names = np.asarray(LABELS, dtype=object)[predictions["anchor_predicted"]]
    subtype_names = np.asarray(SUBTYPE_LABELS, dtype=object)[
        predictions["subtype_predicted"]
    ]
    metadata_subtype_names = np.asarray(SUBTYPE_LABELS, dtype=object)[
        predictions["metadata_subtype_predicted"]
    ]
    probability = predictions["probabilities"]
    prediction_frame = pd.DataFrame(
        {
            "event_id": valid_event_ids,
            "true_label": actual_names,
            "pred_label": predicted_names,
            "anchor_pred_label": anchor_names,
            "prob_benign": probability[:, 0],
            "prob_malicious": probability[:, 1],
            "prob_suspicious": probability[:, 2],
            "threat_probability": outputs["threat"][:, 1],
            "anchor_threat_probability": outputs["anchor_threat"][:, 1],
            "metadata_threat_probability": outputs["metadata_threat"][:, 1],
            "content_threat_probability": outputs["content_threat"][:, 1],
            "subtype_pred_label": subtype_names,
            "metadata_subtype_pred_label": metadata_subtype_names,
            "semantic_combo_count": outputs["combo_counts"].astype(np.int64),
            "novelty_gate": outputs["gate"],
            "conflict_candidate": outputs["conflict"].astype(np.int8),
            "metadata_reliability_candidate": outputs[
                "metadata_candidate"
            ].astype(np.int8),
            "trust_logit": outputs["trust_logit"],
            "trust_score": outputs["trust"],
            "delta_margin": outputs["delta"],
        }
    )
    prediction_frame.to_parquet(
        args.output_dir / "valid_predictions.parquet", index=False
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
