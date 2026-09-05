from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat import LABELS  # noqa: E402
from soc_threat.metrics import evaluate_predictions  # noqa: E402
from soc_threat.text_specialist import (  # noqa: E402
    DEFAULT_RULE_PROFILE,
    SPECIALIST_LABELS,
    best_binary_macro_f1_threshold,
    best_competition_threshold,
    force_rule_probabilities,
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_route_frames(
    train_path: Path,
    valid_input_path: Path,
    valid_answer_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = duckdb.connect()
    connection.execute("SET threads TO 16")
    train = connection.execute(
        f"""
        SELECT event_id, message_sanitized, label_binary
        FROM read_parquet('{sql_path(train_path)}')
        WHERE pipeline = 'syslog' AND COALESCE(product_name, '') = ''
          AND label_binary IN ('benign', 'malicious')
        """
    ).fetchdf()
    valid = connection.execute(
        f"""
        SELECT i.event_id, i.message_sanitized, a.label_binary
        FROM read_parquet('{sql_path(valid_input_path)}') AS i
        JOIN read_parquet('{sql_path(valid_answer_path)}') AS a USING (event_id)
        WHERE i.pipeline = 'syslog' AND COALESCE(i.product_name, '') = ''
          AND a.label_binary IN ('benign', 'malicious')
        """
    ).fetchdf()
    connection.close()
    return train, valid


def load_fixed_confusion(
    base_predictions_path: Path,
    specialist_valid: pd.DataFrame,
) -> np.ndarray:
    connection = duckdb.connect()
    route_ids = pd.DataFrame(
        {"event_id": specialist_valid["event_id"].astype(str).tolist()},
        dtype=object,
    )
    connection.register("specialist_route_ids", route_ids)
    base = sql_path(base_predictions_path)
    grouped = connection.execute(
        f"""
        SELECT b.true_label, b.pred_label, COUNT(*) AS rows
        FROM read_parquet('{base}') AS b
        ANTI JOIN specialist_route_ids AS r USING (event_id)
        GROUP BY b.true_label, b.pred_label
        """
    ).fetchall()
    audit = connection.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM read_parquet('{base}')) AS base_rows,
            (SELECT COUNT(*) FROM specialist_route_ids) AS route_rows,
            (SELECT COUNT(*) FROM read_parquet('{base}') AS b
             JOIN specialist_route_ids AS r USING (event_id)) AS matched_route_rows
        """
    ).fetchone()
    connection.close()
    if int(audit[1]) != int(audit[2]):
        raise ValueError(
            "Some specialist validation event_id values are missing from base predictions"
        )
    if int(audit[0]) < int(audit[1]):
        raise ValueError("Base prediction row count is smaller than specialist route")
    positions = {label: index for index, label in enumerate(LABELS)}
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for true_label, pred_label, rows in grouped:
        if true_label not in positions or pred_label not in positions:
            raise ValueError(
                f"Unknown base prediction labels: true={true_label}, pred={pred_label}"
            )
        matrix[positions[true_label], positions[pred_label]] = int(rows)
    if int(matrix.sum()) + len(specialist_valid) != int(audit[0]):
        raise ValueError("Base predictions contain duplicate or unmatched event_id values")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF specialist for the mixed syslog cluster"
    )
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--valid-answer", type=Path, required=True)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        help="Full v1.0 validation predictions for official-score threshold tuning",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v2_2_text_specialist",
    )
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--rules-profile", choices=("basic", "expanded"), default=DEFAULT_RULE_PROFILE)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.train_raw, args.valid_input, args.valid_answer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.base_predictions is not None and not args.base_predictions.is_file():
        raise FileNotFoundError(args.base_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    train, valid = load_route_frames(
        args.train_raw, args.valid_input, args.valid_answer
    )
    train_text = train["message_sanitized"].fillna("").astype(str)
    valid_text = valid["message_sanitized"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9999,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b[\w.-]{2,}\b",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=1e-4,
        random_state=args.seed,
        average=True,
        n_jobs=-1,
    )
    classifier.fit(train_matrix, train["label_binary"])
    del train_matrix

    valid_matrix = vectorizer.transform(valid_text)
    malicious_index = list(classifier.classes_).index("malicious")
    raw_probabilities = classifier.predict_proba(valid_matrix)[:, malicious_index]
    malicious_probabilities, strong_rules = force_rule_probabilities(
        raw_probabilities, valid_text, profile=args.rules_profile
    )
    actual_labels = valid["label_binary"].to_numpy(dtype=object)
    if args.base_predictions is None:
        threshold_score, threshold = best_binary_macro_f1_threshold(
            malicious_probabilities, actual_labels
        )
        threshold_selection_metric = "specialist_binary_macro_f1"
    else:
        fixed_confusion = load_fixed_confusion(args.base_predictions, valid)
        threshold_score, threshold = best_competition_threshold(
            malicious_probabilities,
            actual_labels,
            fixed_confusion,
        )
        threshold_selection_metric = "official_competition_score"
    predicted_labels = np.where(
        malicious_probabilities >= threshold, "malicious", "benign"
    )
    probability_matrix = np.column_stack(
        [1.0 - malicious_probabilities, malicious_probabilities]
    )
    metrics = evaluate_predictions(
        actual_labels,
        predicted_labels,
        labels=SPECIALIST_LABELS,
        probabilities=probability_matrix,
    )
    metrics.update(
        {
            "model": "v2.1_tfidf_sgd_semantic_rule_specialist",
            "model_version": "v2.1",
            "rules_profile": args.rules_profile,
            "route": "pipeline == syslog and product_name is empty",
            "threshold": threshold,
            "threshold_selection_metric": threshold_selection_metric,
            "threshold_selection_score": threshold_score,
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "strong_rule_rows": int(strong_rules.sum()),
            "strong_rule_malicious_rows": int(
                np.sum(strong_rules & (actual_labels == "malicious"))
            ),
            "strong_rule_benign_rows": int(
                np.sum(strong_rules & (actual_labels == "benign"))
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "arguments": {
                **vars(args),
                "train_raw": str(args.train_raw),
                "valid_input": str(args.valid_input),
                "valid_answer": str(args.valid_answer),
                "base_predictions": (
                    str(args.base_predictions)
                    if args.base_predictions is not None
                    else None
                ),
                "output_dir": str(args.output_dir),
            },
        }
    )
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "rules_profile": args.rules_profile,
            "classifier": classifier,
            "threshold": threshold,
            "threshold_selection_metric": threshold_selection_metric,
            "threshold_selection_score": threshold_score,
            "labels": SPECIALIST_LABELS,
            "route": "pipeline == syslog and product_name is empty",
        },
        args.output_dir / "model.joblib",
        compress=3,
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_table = pa.table(
        {
            "event_id": valid["event_id"].astype(str).tolist(),
            "true_label": actual_labels.tolist(),
            "pred_label": predicted_labels.tolist(),
            "prob_benign": probability_matrix[:, 0].astype(np.float32),
            "prob_malicious": probability_matrix[:, 1].astype(np.float32),
            "strong_rule": strong_rules,
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
