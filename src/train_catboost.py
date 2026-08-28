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
from catboost import CatBoostClassifier, Pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.feature_schema import (  # noqa: E402
    CATEGORICAL_FEATURES as V1_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES as V1_NUMERIC_FEATURES,
)
from soc_threat.metrics import evaluate_predictions  # noqa: E402


def read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def stratified_sample(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Sample across the whole file while preserving all three labels.

    The original files are ordered in a label-dependent way, so taking the
    first N rows would produce a misleading smoke test.
    """

    if max_rows is None or max_rows >= len(frame):
        return frame
    fractions = frame["label_binary"].value_counts(normalize=True)
    parts: list[pd.DataFrame] = []
    remaining = max_rows
    labels = list(fractions.index)
    for index, label in enumerate(labels):
        group = frame.loc[frame["label_binary"] == label]
        if index == len(labels) - 1:
            take = remaining
        else:
            take = max(1, int(round(max_rows * float(fractions[label]))))
            take = min(take, len(group), remaining - (len(labels) - index - 1))
        parts.append(group.sample(n=take, random_state=20260828 + index))
        remaining -= take
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=20260828)
        .reset_index(drop=True)
    )


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    for column in V1_CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("__MISSING__").astype(str)
    for column in V1_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V1 CatBoost baseline")
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_train.parquet",
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_valid.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v1_catboost",
    )
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--thread-count", type=int, default=16)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = V1_CATEGORICAL_FEATURES + V1_NUMERIC_FEATURES
    required = ["event_id", "label_binary", *features]
    for path in (args.train, args.valid):
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    train = stratified_sample(read_parquet(args.train, required), args.max_train_rows)
    valid = stratified_sample(read_parquet(args.valid, required), args.max_valid_rows)
    train = normalize_features(train)
    valid = normalize_features(valid)
    data_load_seconds = time.perf_counter() - started

    train_pool = Pool(
        train[features],
        label=train["label_binary"],
        cat_features=V1_CATEGORICAL_FEATURES,
        feature_names=features,
    )
    valid_pool = Pool(
        valid[features],
        label=valid["label_binary"],
        cat_features=V1_CATEGORICAL_FEATURES,
        feature_names=features,
    )
    model = CatBoostClassifier(
        loss_function="MultiClass",
        # Class weights are needed during learning, but validation Macro-F1 must
        # describe real rows rather than reweighted rows.
        eval_metric="TotalF1:average=Macro;use_weights=false",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=5.0,
        random_seed=20260828,
        auto_class_weights="Balanced",
        random_strength=0.5,
        thread_count=args.thread_count,
        allow_writing_files=False,
        verbose=25,
    )
    train_started = time.perf_counter()
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    train_seconds = time.perf_counter() - train_started

    probabilities_model_order = model.predict_proba(valid_pool)
    model_labels = [str(value) for value in model.classes_]
    label_positions = [model_labels.index(label) for label in LABELS]
    probabilities = probabilities_model_order[:, label_positions]
    prediction = np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    metrics = evaluate_predictions(
        valid["label_binary"],
        prediction,
        labels=LABELS,
        probabilities=probabilities,
    )
    metrics.update(
        {
            "model": "v1_catboost_structured_no_absolute_time_no_raw_ids",
            "features": features,
            "categorical_features": V1_CATEGORICAL_FEATURES,
            "best_iteration": int(model.get_best_iteration()),
            "model_class_order": model_labels,
            "data_load_seconds": data_load_seconds,
            "train_seconds": train_seconds,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "parameters": model.get_params(),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output_dir / "model.cbm")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(train_pool),
        }
    ).sort_values("importance", ascending=False)
    importance.to_parquet(
        args.output_dir / "feature_importance.parquet",
        index=False,
    )

    prediction_table = pa.table(
        {
            "event_id": valid["event_id"].astype(str).tolist(),
            "true_label": valid["label_binary"].astype(str).tolist(),
            "pred_label": prediction.tolist(),
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
