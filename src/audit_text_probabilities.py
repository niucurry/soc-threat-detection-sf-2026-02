from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.text_specialist import strong_malicious_rules  # noqa: E402
from train_text_specialist import load_route_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V2 raw text probabilities across train and validation"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-raw", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument("--valid-answer", type=Path, required=True)
    parser.add_argument("--comparison-threshold", type=float, default=0.44)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v3_generalization/probability_audit.json"),
    )
    return parser.parse_args()


def normalize_family(messages: pd.Series, characters: int = 80) -> pd.Series:
    return (
        messages.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[0-9a-f]{16,}", "<hex>", regex=True)
        .str.replace(r"[0-9]+", "<n>", regex=True)
        .str.slice(0, characters)
    )


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"rows": 0, "min": None, "max": None}
    points = np.quantile(values, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    names = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {"rows": int(len(values)), **dict(zip(names, (float(x) for x in points)))}


def probability_gap_candidates(
    values: np.ndarray,
    min_positive_fraction: float = 0.001,
    max_positive_fraction: float = 0.25,
    limit: int = 20,
) -> list[dict[str, float | int]]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return []
    ordered = np.sort(values)
    gaps = ordered[1:] - ordered[:-1]
    positive_rows = len(values) - np.arange(1, len(values))
    positive_fraction = positive_rows / len(values)
    eligible = (
        (positive_fraction >= min_positive_fraction)
        & (positive_fraction <= max_positive_fraction)
        & (gaps > 0)
    )
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return []
    ranked = indices[np.argsort(gaps[indices])[::-1]][:limit]
    return [
        {
            "threshold": float((ordered[index] + ordered[index + 1]) / 2.0),
            "gap": float(gaps[index]),
            "lower": float(ordered[index]),
            "upper": float(ordered[index + 1]),
            "predicted_positive_rows": int(positive_rows[index]),
            "predicted_positive_fraction": float(positive_fraction[index]),
        }
        for index in ranked
    ]


def probability_groups(
    name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    rules: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = frame["label_binary"].to_numpy(dtype=object)
    result: dict[str, Any] = {"dataset": name, "threshold": threshold, "groups": {}}
    for label in ("benign", "malicious"):
        for rule_value in (False, True):
            mask = (labels == label) & (rules == rule_value)
            key = f"{label}_rule_{str(rule_value).lower()}"
            result["groups"][key] = {
                **quantiles(probabilities[mask]),
                "predicted_malicious": int(np.sum(probabilities[mask] >= threshold)),
            }
    return result


def model_probabilities(package: dict[str, Any], messages: pd.Series) -> np.ndarray:
    matrix = package["vectorizer"].transform(messages.fillna("").astype(str))
    classifier = package["classifier"]
    malicious_index = list(classifier.classes_).index("malicious")
    return classifier.predict_proba(matrix)[:, malicious_index]


def top_families(
    messages: pd.Series,
    probabilities: np.ndarray,
    mask: np.ndarray,
    limit: int = 30,
) -> list[dict[str, Any]]:
    selected = pd.DataFrame(
        {
            "family": normalize_family(messages.loc[mask].reset_index(drop=True)),
            "probability": probabilities[mask],
        }
    )
    if selected.empty:
        return []
    grouped = (
        selected.groupby("family", as_index=False)
        .agg(
            rows=("probability", "size"),
            probability_min=("probability", "min"),
            probability_median=("probability", "median"),
            probability_max=("probability", "max"),
        )
        .sort_values("rows", ascending=False)
        .head(limit)
    )
    return grouped.to_dict(orient="records")


def representative_samples(
    messages: pd.Series,
    probabilities: np.ndarray,
    mask: np.ndarray,
    limit: int = 30,
) -> list[dict[str, Any]]:
    selected_messages = messages.loc[mask].reset_index(drop=True)
    if selected_messages.empty:
        return []
    selected = pd.DataFrame(
        {
            "family": normalize_family(selected_messages),
            "message": selected_messages.str.slice(0, 600),
            "probability": probabilities[mask],
        }
    )
    samples = (
        selected.sort_values("probability")
        .drop_duplicates("family")
        .head(limit)
    )
    return samples.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    for path in (args.model, args.train_raw, args.valid_input, args.valid_answer):
        if not path.is_file():
            raise FileNotFoundError(path)
    package = joblib.load(args.model)
    train, valid = load_route_frames(
        args.train_raw,
        args.valid_input,
        args.valid_answer,
    )
    train_messages = train["message_sanitized"].fillna("").astype(str)
    valid_messages = valid["message_sanitized"].fillna("").astype(str)
    train_probabilities = model_probabilities(package, train_messages)
    valid_probabilities = model_probabilities(package, valid_messages)
    train_rules = strong_malicious_rules(train_messages)
    valid_rules = strong_malicious_rules(valid_messages)
    threshold = float(package["threshold"])

    valid_labels = valid["label_binary"].to_numpy(dtype=object)
    train_labels = train["label_binary"].to_numpy(dtype=object)
    missed_at_comparison = (
        (valid_labels == "malicious")
        & (~valid_rules)
        & (valid_probabilities < args.comparison_threshold)
    )
    benign_labels = valid_labels == "benign"
    report = {
        "model": str(args.model.resolve()),
        "model_threshold": threshold,
        "comparison_threshold": args.comparison_threshold,
        "train": probability_groups(
            "train", train, train_probabilities, train_rules, threshold
        ),
        "valid": probability_groups(
            "valid", valid, valid_probabilities, valid_rules, threshold
        ),
        "valid_malicious_non_rule_below_comparison": top_families(
            valid_messages,
            valid_probabilities,
            missed_at_comparison,
        ),
        "valid_highest_probability_benign_families": top_families(
            valid_messages,
            valid_probabilities,
            benign_labels,
        ),
        "train_non_rule_probability_gaps": probability_gap_candidates(
            train_probabilities[~train_rules]
        ),
        "valid_non_rule_probability_gaps": probability_gap_candidates(
            valid_probabilities[~valid_rules]
        ),
        "train_malicious_non_rule_samples": representative_samples(
            train_messages,
            train_probabilities,
            (train_labels == "malicious") & (~train_rules),
        ),
        "valid_malicious_non_rule_samples": representative_samples(
            valid_messages,
            valid_probabilities,
            (valid_labels == "malicious") & (~valid_rules),
        ),
        "valid_high_probability_benign_samples": representative_samples(
            valid_messages,
            valid_probabilities,
            benign_labels & (valid_probabilities >= np.quantile(
                valid_probabilities[benign_labels], 0.99
            )),
        ),
    }
    classifier = package["classifier"]
    feature_names = package["vectorizer"].get_feature_names_out()
    malicious_index = list(classifier.classes_).index("malicious")
    if len(classifier.classes_) == 2:
        coefficients = classifier.coef_[0]
        if malicious_index == 0:
            coefficients = -coefficients
    else:
        coefficients = classifier.coef_[malicious_index]
    top_positive = np.argsort(coefficients)[-50:][::-1]
    top_negative = np.argsort(coefficients)[:50]
    report["top_positive_features"] = [
        {"feature": str(feature_names[index]), "weight": float(coefficients[index])}
        for index in top_positive
    ]
    report["top_negative_features"] = [
        {"feature": str(feature_names[index]), "weight": float(coefficients[index])}
        for index in top_negative
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
