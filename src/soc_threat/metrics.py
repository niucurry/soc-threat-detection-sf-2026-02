from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable multiclass metrics.

    Macro-F1 is the primary local metric: each class contributes equally even
    though benign rows are much more frequent.
    """

    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true={true.shape}, pred={pred.shape}")
    true_indices = np.full(len(true), -1, dtype=np.int64)
    pred_indices = np.full(len(pred), -1, dtype=np.int64)
    for index, label in enumerate(labels):
        true_indices[true == label] = index
        pred_indices[pred == label] = index
    if np.any(true_indices < 0) or np.any(pred_indices < 0):
        raise ValueError("Unknown label encountered during evaluation")

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(matrix, (true_indices, pred_indices), 1)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    supports = matrix.sum(axis=1)
    predicted_totals = matrix.sum(axis=0)
    true_positives = np.diag(matrix).astype(float)
    precisions = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros(len(labels), dtype=float),
        where=predicted_totals != 0,
    )
    recalls = np.divide(
        true_positives,
        supports,
        out=np.zeros(len(labels), dtype=float),
        where=supports != 0,
    )
    f1_values = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(len(labels), dtype=float),
        where=(precisions + recalls) != 0,
    )
    result: dict[str, Any] = {
        "rows": int(len(true)),
        "accuracy": float(true_positives.sum() / max(1, len(true))),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(f1_values.mean()),
        "weighted_f1": float(
            np.average(f1_values, weights=supports) if supports.sum() else 0.0
        ),
        "per_class": {
            label: {
                "precision": float(precisions[index]),
                "recall": float(recalls[index]),
                "f1": float(f1_values[index]),
                "support": int(supports[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "counts": matrix.astype(int).tolist(),
            "row_normalized": normalized.tolist(),
        },
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(true), len(labels)):
            raise ValueError(
                f"Probability shape must be {(len(true), len(labels))}, "
                f"got {probabilities.shape}"
            )
        selected = probabilities[np.arange(len(true)), true_indices]
        result["multiclass_log_loss"] = float(
            -np.log(np.clip(selected, 1e-15, 1.0)).mean()
        )
    return result
