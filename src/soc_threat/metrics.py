from __future__ import annotations

from typing import Any, Iterable

import numpy as np


COMPETITION_WEIGHTS = {
    "threat_binary_f1": 0.40,
    "threat_binary_recall": 0.25,
    "threat_recall": 0.15,
    "macro_f1": 0.10,
    "soft_label_score": 0.05,
    "balanced_accuracy": 0.05,
}


def competition_metrics_from_confusion(
    matrix: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Calculate the six weighted metrics shown in the official scoring slide.

    Soft Label Score is interpreted as the row-wise mean of: exact match=1,
    malicious/suspicious cross-confusion=0.5, and normal/threat confusion=0.
    """

    required = {"benign", "malicious", "suspicious"}
    if set(labels) != required or len(labels) != 3:
        raise ValueError(
            "Competition metrics require benign, malicious, and suspicious labels"
        )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Competition confusion matrix must be 3x3, got {matrix.shape}")
    positions = {label: labels.index(label) for label in required}
    benign = positions["benign"]
    malicious = positions["malicious"]
    suspicious = positions["suspicious"]

    threat_true_positive = (
        matrix[malicious, malicious]
        + matrix[malicious, suspicious]
        + matrix[suspicious, malicious]
        + matrix[suspicious, suspicious]
    )
    threat_false_positive = matrix[benign, malicious] + matrix[benign, suspicious]
    threat_false_negative = matrix[malicious, benign] + matrix[suspicious, benign]
    threat_precision = float(
        threat_true_positive
        / max(1.0, threat_true_positive + threat_false_positive)
    )
    threat_binary_recall = float(
        threat_true_positive
        / max(1.0, threat_true_positive + threat_false_negative)
    )
    threat_binary_f1 = float(
        2.0
        * threat_precision
        * threat_binary_recall
        / max(1e-15, threat_precision + threat_binary_recall)
    )

    supports = matrix.sum(axis=1)
    predicted_totals = matrix.sum(axis=0)
    true_positives = np.diag(matrix)
    recalls = np.divide(
        true_positives,
        supports,
        out=np.zeros(3, dtype=float),
        where=supports != 0,
    )
    precisions = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros(3, dtype=float),
        where=predicted_totals != 0,
    )
    f1_values = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(3, dtype=float),
        where=(precisions + recalls) != 0,
    )
    threat_recall = float((recalls[malicious] + recalls[suspicious]) / 2.0)
    macro_f1 = float(f1_values.mean())
    balanced_accuracy = float(recalls.mean())
    total = float(matrix.sum())
    soft_label_score = float(
        (
            true_positives.sum()
            + 0.5
            * (
                matrix[malicious, suspicious]
                + matrix[suspicious, malicious]
            )
        )
        / max(1.0, total)
    )
    components = {
        "threat_binary_f1": threat_binary_f1,
        "threat_binary_recall": threat_binary_recall,
        "threat_recall": threat_recall,
        "macro_f1": macro_f1,
        "soft_label_score": soft_label_score,
        "balanced_accuracy": balanced_accuracy,
    }
    final_score = float(
        sum(COMPETITION_WEIGHTS[name] * value for name, value in components.items())
    )
    return {
        "competition_score": final_score,
        "competition_metrics": {
            **components,
            "threat_binary_precision": threat_precision,
            "weights": COMPETITION_WEIGHTS,
            "soft_label_definition": (
                "row_mean: exact=1, malicious/suspicious cross=0.5, "
                "benign/threat confusion=0"
            ),
        },
    }


def evaluate_predictions(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    *,
    labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable multiclass metrics.

    For three-class SOC evaluation, the official weighted competition score is
    the primary selection metric. Macro-F1 remains available as one component.
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
    if set(labels) == {"benign", "malicious", "suspicious"} and len(labels) == 3:
        result.update(competition_metrics_from_confusion(matrix, labels))
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
