from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.metrics import evaluate_predictions  # noqa: E402
from soc_threat.text_specialist import (  # noqa: E402
    SPECIALIST_LABELS,
    adaptive_probability_gap_threshold,
    best_binary_macro_f1_threshold,
    force_rule_probabilities,
    strong_malicious_rules,
)
from train_text_specialist import load_route_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V3 text models on grouped generalization holdouts"
    )
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--valid-answer", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v3_text_experiment",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument(
        "--group-mode",
        choices=("template", "family"),
        default="template",
    )
    parser.add_argument("--family-chars", type=int, default=48)
    parser.add_argument("--skip-char", action="store_true")
    parser.add_argument("--normalize-model-text", action="store_true")
    parser.add_argument("--word-features", type=int, default=200_000)
    parser.add_argument("--char-features", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def normalize_template(messages: pd.Series) -> pd.Series:
    return (
        messages.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[0-9a-f]{16,}", "<hex>", regex=True)
        .str.replace(r"[0-9]+", "<n>", regex=True)
    )


def normalize_model_text(messages: pd.Series) -> pd.Series:
    return (
        messages.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            " iptoken ",
            regex=True,
        )
        .str.replace(r"[0-9a-f]{12,}", " hextoken ", regex=True)
        .str.replace(r"[0-9]+", " numtoken ", regex=True)
    )


def template_folds(
    messages: pd.Series,
    folds: int,
    group_mode: str,
    family_chars: int,
) -> tuple[np.ndarray, pd.Series]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    groups = normalize_template(messages)
    if group_mode == "family":
        if family_chars < 8:
            raise ValueError("family-chars must be at least 8")
        groups = groups.str.slice(0, family_chars)
    hashed = pd.util.hash_pandas_object(groups, index=False).to_numpy(
        dtype=np.uint64
    )
    return (hashed % np.uint64(folds)).astype(np.int8), groups


def make_vectorizer(kind: str, max_features: int) -> TfidfVectorizer:
    common: dict[str, Any] = {
        "lowercase": True,
        "min_df": 2,
        "max_df": 0.9999,
        "max_features": max_features,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w.-]{2,}\b",
            **common,
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            **common,
        )
    raise ValueError(f"Unknown vectorizer kind: {kind}")


def fit_probabilities(
    kind: str,
    train_text: pd.Series,
    train_labels: pd.Series,
    holdout_text: pd.Series,
    external_text: pd.Series,
    max_features: int,
    alpha: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    vectorizer = make_vectorizer(kind, max_features)
    train_matrix = vectorizer.fit_transform(train_text)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
        average=True,
        n_jobs=-1,
    )
    classifier.fit(train_matrix, train_labels)
    malicious_index = list(classifier.classes_).index("malicious")
    holdout_probabilities = classifier.predict_proba(
        vectorizer.transform(holdout_text)
    )[:, malicious_index]
    external_probabilities = classifier.predict_proba(
        vectorizer.transform(external_text)
    )[:, malicious_index]
    details = {
        "kind": kind,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_rows": int(len(train_text)),
        "holdout_rows": int(len(holdout_text)),
        "external_rows": int(len(external_text)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    return holdout_probabilities, external_probabilities, details


def score_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = np.where(probabilities >= threshold, "malicious", "benign")
    metrics = evaluate_predictions(
        labels,
        predictions,
        labels=SPECIALIST_LABELS,
        probabilities=np.column_stack([1.0 - probabilities, probabilities]),
    )
    metrics["threshold"] = float(threshold)
    metrics["errors"] = int(np.sum(predictions != labels))
    return metrics


def evaluate_variant(
    name: str,
    holdout_probabilities: np.ndarray,
    external_probabilities: np.ndarray,
    holdout_labels: np.ndarray,
    external_labels: np.ndarray,
    holdout_messages: pd.Series,
    external_messages: pd.Series,
) -> dict[str, Any]:
    raw_score, raw_threshold = best_binary_macro_f1_threshold(
        holdout_probabilities,
        holdout_labels,
    )
    holdout_rule_probabilities, holdout_rules = force_rule_probabilities(
        holdout_probabilities,
        holdout_messages,
    )
    external_rule_probabilities, external_rules = force_rule_probabilities(
        external_probabilities,
        external_messages,
    )
    rule_score, rule_threshold = best_binary_macro_f1_threshold(
        holdout_rule_probabilities,
        holdout_labels,
    )
    adaptive_holdout_threshold, adaptive_holdout_diagnostics = (
        adaptive_probability_gap_threshold(
            holdout_probabilities,
            excluded=holdout_rules,
            max_positive_fraction=0.40,
        )
    )
    adaptive_external_threshold, adaptive_external_diagnostics = (
        adaptive_probability_gap_threshold(
            external_probabilities,
            excluded=external_rules,
        )
    )
    return {
        "name": name,
        "raw": {
            "selection_macro_f1": raw_score,
            "holdout": score_probabilities(
                holdout_probabilities, holdout_labels, raw_threshold
            ),
            "external": score_probabilities(
                external_probabilities, external_labels, raw_threshold
            ),
        },
        "rules": {
            "selection_macro_f1": rule_score,
            "holdout_rule_rows": int(holdout_rules.sum()),
            "external_rule_rows": int(external_rules.sum()),
            "holdout": score_probabilities(
                holdout_rule_probabilities, holdout_labels, rule_threshold
            ),
            "external": score_probabilities(
                external_rule_probabilities, external_labels, rule_threshold
            ),
        },
        "adaptive": {
            "holdout_diagnostics": adaptive_holdout_diagnostics,
            "external_diagnostics": adaptive_external_diagnostics,
            "holdout": score_probabilities(
                holdout_rule_probabilities,
                holdout_labels,
                adaptive_holdout_threshold,
            ),
            "external": score_probabilities(
                external_rule_probabilities,
                external_labels,
                adaptive_external_threshold,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.holdout_fold < 0 or args.holdout_fold >= args.folds:
        raise ValueError("holdout-fold must be in [0, folds)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, external = load_route_frames(
        args.train_raw,
        args.valid_input,
        args.valid_answer,
    )
    folds, groups = template_folds(
        train["message_sanitized"],
        args.folds,
        args.group_mode,
        args.family_chars,
    )
    is_holdout = folds == args.holdout_fold
    train_part = train.loc[~is_holdout].reset_index(drop=True)
    holdout = train.loc[is_holdout].reset_index(drop=True)
    train_groups = set(groups.loc[~is_holdout].unique())
    holdout_groups = set(groups.loc[is_holdout].unique())
    group_overlap = len(train_groups & holdout_groups)
    if group_overlap:
        raise ValueError(f"Group holdout leakage detected: {group_overlap}")

    train_messages = train_part["message_sanitized"].fillna("").astype(str)
    holdout_messages = holdout["message_sanitized"].fillna("").astype(str)
    external_messages = external["message_sanitized"].fillna("").astype(str)
    if args.normalize_model_text:
        train_text = normalize_model_text(train_messages)
        holdout_text = normalize_model_text(holdout_messages)
        external_text = normalize_model_text(external_messages)
    else:
        train_text = train_messages
        holdout_text = holdout_messages
        external_text = external_messages
    holdout_labels = holdout["label_binary"].to_numpy(dtype=object)
    external_labels = external["label_binary"].to_numpy(dtype=object)

    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    model_details: list[dict[str, Any]] = []
    requested_models = [("word", args.word_features)]
    if not args.skip_char:
        requested_models.append(("char", args.char_features))
    for kind, max_features in requested_models:
        holdout_probabilities, external_probabilities, details = fit_probabilities(
            kind,
            train_text,
            train_part["label_binary"],
            holdout_text,
            external_text,
            max_features,
            args.alpha,
            args.max_iter,
            args.seed,
        )
        probabilities[kind] = (holdout_probabilities, external_probabilities)
        model_details.append(details)

    probability_artifact: dict[str, np.ndarray] = {
        "holdout_labels": holdout_labels,
        "external_labels": external_labels,
        "holdout_rules": strong_malicious_rules(holdout_messages),
        "external_rules": strong_malicious_rules(external_messages),
    }
    for kind, (holdout_values, external_values) in probabilities.items():
        probability_artifact[f"{kind}_holdout"] = holdout_values
        probability_artifact[f"{kind}_external"] = external_values
    np.savez_compressed(
        args.output_dir / "probabilities.npz",
        **probability_artifact,
    )

    variants: list[dict[str, Any]] = []
    word_holdout, word_external = probabilities["word"]
    if "char" in probabilities:
        char_holdout, char_external = probabilities["char"]
        blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    else:
        char_holdout = np.zeros_like(word_holdout)
        char_external = np.zeros_like(word_external)
        blend_weights = (0.0,)
    for char_weight in blend_weights:
        word_weight = 1.0 - char_weight
        variants.append(
            evaluate_variant(
                f"word_{word_weight:.2f}_char_{char_weight:.2f}",
                word_weight * word_holdout + char_weight * char_holdout,
                word_weight * word_external + char_weight * char_external,
                holdout_labels,
                external_labels,
                holdout_messages,
                external_messages,
            )
        )

    best_raw = max(
        variants,
        key=lambda item: float(item["raw"]["holdout"]["macro_f1"]),
    )
    best_rules = max(
        variants,
        key=lambda item: float(item["rules"]["holdout"]["macro_f1"]),
    )
    report = {
        "model": "v3_text_generalization_experiment",
        "split": {
            "folds": args.folds,
            "holdout_fold": args.holdout_fold,
            "group_mode": args.group_mode,
            "family_chars": args.family_chars,
            "normalize_model_text": args.normalize_model_text,
            "train_rows": int(len(train_part)),
            "holdout_rows": int(len(holdout)),
            "external_rows": int(len(external)),
            "train_label_counts": {
                str(key): int(value)
                for key, value in train_part["label_binary"].value_counts().items()
            },
            "holdout_label_counts": {
                str(key): int(value)
                for key, value in holdout["label_binary"].value_counts().items()
            },
            "group_overlap": group_overlap,
        },
        "model_details": model_details,
        "variants": variants,
        "best_raw_variant": best_raw["name"],
        "best_rules_variant": best_rules["name"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "arguments": {
            **vars(args),
            "train_raw": str(args.train_raw),
            "valid_input": str(args.valid_input),
            "valid_answer": str(args.valid_answer),
            "output_dir": str(args.output_dir),
        },
    }
    output_path = args.output_dir / "report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
