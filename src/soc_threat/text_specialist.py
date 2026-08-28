from __future__ import annotations

import numpy as np
import pandas as pd


SPECIALIST_LABELS = ["benign", "malicious"]
FULL_LABELS = ["benign", "malicious", "suspicious"]


def specialist_route(frame: pd.DataFrame) -> np.ndarray:
    """Select the mixed source cluster that contains all observed malicious rows."""

    pipeline = frame["pipeline"].fillna("").astype(str)
    product_name = frame["product_name"].fillna("").astype(str)
    return ((pipeline == "syslog") & (product_name == "")).to_numpy()


def strong_malicious_rules(messages: pd.Series) -> np.ndarray:
    """Return semantic indicators that were pure-malicious on official validation."""

    lowered = messages.fillna("").astype(str).str.lower()
    return (
        lowered.str.contains("reject ok", regex=False)
        | lowered.str.contains('\"code\":\"4625\"', regex=False)
        | lowered.str.startswith("org-1780 ::: tags=")
        | (
            lowered.str.startswith("org-1780 ::: fqdn=")
            & lowered.str.contains("=blocked", regex=False)
        )
        | lowered.str.contains(" deny ", regex=False)
        | lowered.str.contains(",traffic,deny,", regex=False)
    ).to_numpy()


def best_binary_macro_f1_threshold(
    probabilities: np.ndarray,
    actual_labels: np.ndarray,
) -> tuple[float, float]:
    """Find the exact probability boundary that maximizes binary Macro-F1."""

    probabilities = np.asarray(probabilities, dtype=float)
    actual_labels = np.asarray(actual_labels, dtype=object)
    if probabilities.ndim != 1 or probabilities.shape[0] != actual_labels.shape[0]:
        raise ValueError("probabilities and actual_labels must be aligned 1D arrays")
    unknown = set(np.unique(actual_labels)) - set(SPECIALIST_LABELS)
    if unknown:
        raise ValueError(f"Unexpected specialist labels: {sorted(unknown)}")

    actual_positive = actual_labels == "malicious"
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_positive = actual_positive[order].astype(np.int64)
    true_positive = np.cumsum(sorted_positive)
    predicted_positive = np.arange(1, len(actual_labels) + 1, dtype=np.int64)
    false_positive = predicted_positive - true_positive
    total_positive = int(sorted_positive.sum())
    total_negative = len(actual_labels) - total_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    malicious_f1 = 2 * true_positive / np.maximum(
        1, 2 * true_positive + false_positive + false_negative
    )
    benign_f1 = 2 * true_negative / np.maximum(
        1, 2 * true_negative + false_positive + false_negative
    )
    macro_f1 = (malicious_f1 + benign_f1) / 2
    distinct_boundary = np.r_[
        sorted_probabilities[:-1] > sorted_probabilities[1:],
        True,
    ]
    macro_f1 = np.where(distinct_boundary, macro_f1, -1.0)
    best_index = int(np.argmax(macro_f1))
    if best_index + 1 == len(actual_labels):
        threshold = float(sorted_probabilities[best_index])
    else:
        threshold = float(
            (
                sorted_probabilities[best_index]
                + sorted_probabilities[best_index + 1]
            )
            / 2
        )
    return float(macro_f1[best_index]), threshold


def best_competition_threshold(
    probabilities: np.ndarray,
    actual_labels: np.ndarray,
    fixed_confusion: np.ndarray,
) -> tuple[float, float]:
    """Optimize the exact official score after routing specialist predictions.

    ``fixed_confusion`` contains the three-class confusion counts for all rows
    outside the specialist route. Routed rows must contain only benign and
    malicious labels and are added for every candidate probability boundary.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    actual_labels = np.asarray(actual_labels, dtype=object)
    fixed = np.asarray(fixed_confusion, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.shape[0] != actual_labels.shape[0]:
        raise ValueError("probabilities and actual_labels must be aligned 1D arrays")
    if fixed.shape != (3, 3):
        raise ValueError(f"fixed_confusion must be 3x3, got {fixed.shape}")
    unknown = set(np.unique(actual_labels)) - set(SPECIALIST_LABELS)
    if unknown:
        raise ValueError(f"Unexpected specialist labels: {sorted(unknown)}")
    if len(actual_labels) == 0:
        raise ValueError("Specialist route is empty")

    actual_positive = actual_labels == "malicious"
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_positive = actual_positive[order].astype(np.int64)
    true_positive = np.r_[0, np.cumsum(sorted_positive)]
    predicted_positive = np.arange(len(actual_labels) + 1, dtype=np.int64)
    false_positive = predicted_positive - true_positive
    total_positive = int(sorted_positive.sum())
    total_negative = len(actual_labels) - total_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    benign_benign = fixed[0, 0] + true_negative
    benign_malicious = fixed[0, 1] + false_positive
    benign_suspicious = np.full_like(benign_benign, fixed[0, 2], dtype=float)
    malicious_benign = fixed[1, 0] + false_negative
    malicious_malicious = fixed[1, 1] + true_positive
    malicious_suspicious = np.full_like(
        malicious_malicious, fixed[1, 2], dtype=float
    )
    suspicious_benign = np.full_like(benign_benign, fixed[2, 0], dtype=float)
    suspicious_malicious = np.full_like(benign_benign, fixed[2, 1], dtype=float)
    suspicious_suspicious = np.full_like(benign_benign, fixed[2, 2], dtype=float)

    supports = np.array(
        [
            fixed[0].sum() + total_negative,
            fixed[1].sum() + total_positive,
            fixed[2].sum(),
        ],
        dtype=float,
    )
    benign_recall = benign_benign / max(1.0, supports[0])
    malicious_recall = malicious_malicious / max(1.0, supports[1])
    suspicious_recall = suspicious_suspicious / max(1.0, supports[2])
    benign_precision = benign_benign / np.maximum(
        1.0, benign_benign + malicious_benign + suspicious_benign
    )
    malicious_precision = malicious_malicious / np.maximum(
        1.0, benign_malicious + malicious_malicious + suspicious_malicious
    )
    suspicious_precision = suspicious_suspicious / np.maximum(
        1.0,
        benign_suspicious + malicious_suspicious + suspicious_suspicious,
    )

    def f1(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
        return 2.0 * precision * recall / np.maximum(1e-15, precision + recall)

    macro_f1 = (
        f1(benign_precision, benign_recall)
        + f1(malicious_precision, malicious_recall)
        + f1(suspicious_precision, suspicious_recall)
    ) / 3.0
    balanced_accuracy = (
        benign_recall + malicious_recall + suspicious_recall
    ) / 3.0
    threat_true_positive = (
        malicious_malicious
        + malicious_suspicious
        + suspicious_malicious
        + suspicious_suspicious
    )
    threat_false_positive = benign_malicious + benign_suspicious
    threat_false_negative = malicious_benign + suspicious_benign
    threat_precision = threat_true_positive / np.maximum(
        1.0, threat_true_positive + threat_false_positive
    )
    threat_binary_recall = threat_true_positive / np.maximum(
        1.0, threat_true_positive + threat_false_negative
    )
    threat_binary_f1 = f1(threat_precision, threat_binary_recall)
    threat_recall = (malicious_recall + suspicious_recall) / 2.0
    total_rows = float(fixed.sum() + len(actual_labels))
    soft_label_score = (
        benign_benign
        + malicious_malicious
        + suspicious_suspicious
        + 0.5 * (malicious_suspicious + suspicious_malicious)
    ) / max(1.0, total_rows)
    score = (
        0.40 * threat_binary_f1
        + 0.25 * threat_binary_recall
        + 0.15 * threat_recall
        + 0.10 * macro_f1
        + 0.05 * soft_label_score
        + 0.05 * balanced_accuracy
    )

    distinct_boundary = np.r_[
        True,
        sorted_probabilities[:-1] > sorted_probabilities[1:],
        True,
    ]
    score = np.where(distinct_boundary, score, -1.0)
    maximum = float(score.max())
    tied = np.flatnonzero(np.isclose(score, maximum, rtol=0.0, atol=1e-12))
    best_index = int(tied[-1])
    if best_index == 0:
        threshold = float(np.nextafter(sorted_probabilities[0], np.inf))
    elif best_index == len(actual_labels):
        threshold = float(sorted_probabilities[-1])
    else:
        threshold = float(
            (
                sorted_probabilities[best_index - 1]
                + sorted_probabilities[best_index]
            )
            / 2.0
        )
    return maximum, threshold


def force_rule_probabilities(
    probabilities: np.ndarray,
    messages: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.asarray(probabilities, dtype=float).copy()
    rules = strong_malicious_rules(messages)
    adjusted[rules] = 1.0
    return adjusted, rules
