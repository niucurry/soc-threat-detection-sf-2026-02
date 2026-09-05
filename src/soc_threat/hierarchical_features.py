from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from soc_threat.feature_schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from train_npu_tabular import fit_preprocessor, transform_inputs


MISSING_CATEGORY = "__MISSING__"
COMBO_SEPARATOR = "\x1f"

# Metadata is intentionally independent of parsed message semantics. The
# subtype fallback can therefore use the source/product prior when the
# semantic combination was never observed in training.
METADATA_CATEGORICAL_FEATURES = [*CATEGORICAL_FEATURES, "vendor_name"]
METADATA_NUMERIC_FEATURES = list(NUMERIC_FEATURES)

SEMANTIC_CATEGORICAL_FEATURES = [
    "content_family",
    "content_action",
    "content_protocol",
    "content_event_code",
]
SEMANTIC_BASE_NUMERIC_FEATURES = [
    "content_has_threat",
    "content_has_authentication",
    "content_has_potentially_harmful",
    "raw_token_count",
]
SEMANTIC_NUMERIC_FEATURES = list(SEMANTIC_BASE_NUMERIC_FEATURES)

NOVELTY_COMBO_COLUMNS = [
    "product_name",
    "content_family",
    "content_action",
]

HIERARCHICAL_REQUIRED_COLUMNS = list(
    dict.fromkeys(
        [
            *METADATA_CATEGORICAL_FEATURES,
            *METADATA_NUMERIC_FEATURES,
            *SEMANTIC_CATEGORICAL_FEATURES,
            *SEMANTIC_BASE_NUMERIC_FEATURES,
            *NOVELTY_COMBO_COLUMNS,
        ]
    )
)


@dataclass(frozen=True)
class HierarchicalArrays:
    metadata_categorical: np.ndarray
    metadata_numeric: np.ndarray
    semantic_categorical: np.ndarray
    semantic_numeric: np.ndarray
    novelty_gate: np.ndarray
    combo_counts: np.ndarray


def _normalized_text(frame: pd.DataFrame, column: str) -> pd.Series:
    return (
        frame[column].fillna(MISSING_CATEGORY).astype(str).replace("", MISSING_CATEGORY)
    )


def semantic_combo_keys(frame: pd.DataFrame) -> pd.Series:
    key = _normalized_text(frame, NOVELTY_COMBO_COLUMNS[0])
    for column in NOVELTY_COMBO_COLUMNS[1:]:
        key = key.str.cat(_normalized_text(frame, column), sep=COMBO_SEPARATOR)
    return key


def _semantic_frame(
    frame: pd.DataFrame,
    combo_counts: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    selected = [*SEMANTIC_CATEGORICAL_FEATURES, *SEMANTIC_BASE_NUMERIC_FEATURES]
    semantic = frame[selected].copy()
    counts = (
        semantic_combo_keys(frame)
        .map(combo_counts)
        .fillna(0)
        .to_numpy(dtype=np.float32, copy=True)
    )
    return semantic, counts


def fit_hierarchical_preprocessor(
    train: pd.DataFrame,
    *,
    novelty_pseudocount: float = 32.0,
) -> dict[str, Any]:
    if novelty_pseudocount <= 0:
        raise ValueError("novelty_pseudocount must be positive")
    missing = [name for name in HIERARCHICAL_REQUIRED_COLUMNS if name not in train]
    if missing:
        raise ValueError(f"Missing hierarchical feature columns: {missing}")

    counts = semantic_combo_keys(train).value_counts(dropna=False)
    combo_counts = {str(key): int(value) for key, value in counts.items()}
    semantic, _ = _semantic_frame(train, combo_counts)
    return {
        "format_version": 1,
        "model_version": "v4.0",
        "metadata": fit_preprocessor(
            train,
            METADATA_CATEGORICAL_FEATURES,
            METADATA_NUMERIC_FEATURES,
        ),
        "semantic": fit_preprocessor(
            semantic,
            SEMANTIC_CATEGORICAL_FEATURES,
            SEMANTIC_NUMERIC_FEATURES,
        ),
        "combo_columns": NOVELTY_COMBO_COLUMNS,
        "combo_counts": combo_counts,
        "novelty_pseudocount": float(novelty_pseudocount),
        "leakage_guard": (
            "Combination counts use training input fields only; labels and validation "
            "rows are never used"
        ),
    }


def transform_hierarchical_inputs(
    frame: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> HierarchicalArrays:
    combo_counts = {
        str(key): int(value) for key, value in preprocessor["combo_counts"].items()
    }
    semantic, counts = _semantic_frame(frame, combo_counts)
    metadata_categorical, metadata_numeric = transform_inputs(
        frame, preprocessor["metadata"]
    )
    semantic_categorical, semantic_numeric = transform_inputs(
        semantic, preprocessor["semantic"]
    )
    pseudocount = float(preprocessor["novelty_pseudocount"])
    gate = counts / (counts + pseudocount)
    return HierarchicalArrays(
        metadata_categorical=metadata_categorical,
        metadata_numeric=metadata_numeric,
        semantic_categorical=semantic_categorical,
        semantic_numeric=semantic_numeric,
        novelty_gate=gate.astype(np.float32, copy=False),
        combo_counts=counts.astype(np.int64, copy=False),
    )
