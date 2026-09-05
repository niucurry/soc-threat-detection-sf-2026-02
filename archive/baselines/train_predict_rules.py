"""SOC 日志三分类：高置信规则学习基线。

规则完全从训练集统计得到，不使用验证答案训练。验证答案仅用于本地评分。
适用于当前约 200 万行训练、200 万行测试数据，运行时不构造高维稀疏矩阵。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


LABELS = ("benign", "suspicious", "malicious")
DEFAULT_CATEGORICAL_COLUMNS = ("product_name", "vendor_name", "pipeline")
DEFAULT_NULL_COLUMNS = ("dst_host", "username", "src_ip", "dst_ip", "src_port")

# Windows 终端在中文用户名环境下有时继承错误代码页，统一为 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Rule:
    kind: str
    column: str
    value: str | None
    pred_label: str
    support: int
    purity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOC 威胁检测高置信规则基线")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "data",
        help="原始 Parquet 数据目录（默认使用仓库内的 data/data）",
    )
    parser.add_argument("--train-file", default="train.parquet")
    parser.add_argument("--test-file", default="valid_input.parquet")
    parser.add_argument("--answer-file", default="valid_answer_private.parquet")
    parser.add_argument("--output", type=Path, default=Path("res.csv"))
    parser.add_argument("--rules-output", type=Path, default=Path("artifacts/rules.json"))
    parser.add_argument("--id-col", default="event_id")
    parser.add_argument("--label-col", default="label_binary")
    parser.add_argument("--min-support", type=int, default=500)
    parser.add_argument("--min-purity", type=float, default=0.98)
    parser.add_argument(
        "--categorical-columns",
        nargs="*",
        default=list(DEFAULT_CATEGORICAL_COLUMNS),
    )
    parser.add_argument(
        "--null-columns",
        nargs="*",
        default=list(DEFAULT_NULL_COLUMNS),
    )
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def validate_train(train: pd.DataFrame, id_col: str, label_col: str) -> None:
    required = {id_col, label_col}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"训练集缺少字段: {sorted(missing)}")
    observed = set(train[label_col].dropna().astype(str).str.lower().unique())
    unknown = observed - set(LABELS)
    if unknown:
        raise ValueError(f"训练集存在未知标签: {sorted(unknown)}")
    if train[id_col].isna().any() or train[id_col].duplicated().any():
        raise ValueError("训练集 event_id 存在空值或重复")


def validate_test(test: pd.DataFrame, id_col: str) -> None:
    if id_col not in test:
        raise ValueError(f"测试集缺少字段: {id_col}")
    if test[id_col].isna().any() or test[id_col].duplicated().any():
        raise ValueError("测试集 event_id 存在空值或重复")


def learn_rules(
    train: pd.DataFrame,
    label_col: str,
    categorical_columns: list[str],
    null_columns: list[str],
    min_support: int,
    min_purity: float,
) -> list[Rule]:
    """学习少数类高置信规则，默认 benign，避免多数类规则淹没结果。"""
    rules: list[Rule] = []
    y = train[label_col].astype("string").str.lower()

    # 缺失值和空字符串必须区分；本数据中二者语义不同。
    for column in null_columns:
        if column not in train:
            continue
        mask = train[column].isna()
        support = int(mask.sum())
        if support < min_support:
            continue
        counts = y[mask].value_counts()
        pred_label = str(counts.index[0])
        purity = float(counts.iloc[0] / support)
        if pred_label != "benign" and purity >= min_purity:
            rules.append(Rule("is_null", column, None, pred_label, support, purity))

    for column in categorical_columns:
        if column not in train:
            continue
        values = train[column].astype("string").fillna("__NULL__")
        stats = (
            pd.DataFrame({"value": values, "label": y})
            .groupby(["value", "label"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        supports = stats.sum(axis=1)
        winners = stats.idxmax(axis=1)
        purities = stats.max(axis=1) / supports
        selected = stats.index[
            (supports >= min_support)
            & (purities >= min_purity)
            & (winners != "benign")
        ]
        for value in selected:
            rules.append(
                Rule(
                    "equals",
                    column,
                    str(value),
                    str(winners.loc[value]),
                    int(supports.loc[value]),
                    float(purities.loc[value]),
                )
            )

    # 先应用类别规则，最后应用纯度更高的空值规则；同类按纯度和支持度排序。
    rules.sort(key=lambda r: (r.kind == "is_null", r.purity, r.support))
    return rules


def predict(test: pd.DataFrame, rules: list[Rule]) -> tuple[np.ndarray, dict[str, int]]:
    pred = np.full(len(test), "benign", dtype=object)
    hits: dict[str, int] = {}
    for i, rule in enumerate(rules, start=1):
        if rule.column not in test:
            hits[f"rule_{i}"] = 0
            continue
        if rule.kind == "is_null":
            mask = test[rule.column].isna().to_numpy()
        elif rule.kind == "equals":
            values = test[rule.column].astype("string").fillna("__NULL__")
            mask = values.eq(rule.value).to_numpy()
        else:
            raise ValueError(f"未知规则类型: {rule.kind}")
        pred[mask] = rule.pred_label
        hits[f"rule_{i}"] = int(mask.sum())
    return pred, hits


def evaluate(
    result: pd.DataFrame,
    answer_path: Path,
    id_col: str,
    label_col: str,
) -> None:
    answer = pd.read_parquet(answer_path, columns=[id_col, label_col])
    if answer[id_col].duplicated().any():
        raise ValueError("答案文件 event_id 存在重复")
    scored = result.merge(answer, on=id_col, how="left", validate="one_to_one")
    if scored[label_col].isna().any():
        raise ValueError("部分预测 event_id 在答案文件中不存在")
    y_true = scored[label_col].astype(str).str.lower()
    y_pred = scored["pred_label"]
    score = f1_score(y_true, y_pred, labels=list(LABELS), average="macro")
    print(f"\n验证集 macro-F1: {score:.9f}")
    print(classification_report(y_true, y_pred, labels=list(LABELS), digits=6, zero_division=0))
    print("混淆矩阵（行=真实，列=预测；顺序 benign/suspicious/malicious）：")
    print(confusion_matrix(y_true, y_pred, labels=list(LABELS)))


def main() -> None:
    args = parse_args()
    started = time.time()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file
    answer_path = data_dir / args.answer_file

    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.5 <= args.min_purity <= 1.0:
        raise ValueError("--min-purity 必须位于 [0.5, 1.0]")
    if args.min_support < 1:
        raise ValueError("--min-support 必须大于 0")

    print(f"读取训练集: {train_path}")
    train = pd.read_parquet(train_path)
    print(f"读取测试集: {test_path}")
    test = pd.read_parquet(test_path)
    validate_train(train, args.id_col, args.label_col)
    validate_test(test, args.id_col)
    train[args.label_col] = train[args.label_col].astype("string").str.lower()
    print("训练标签分布:", train[args.label_col].value_counts().to_dict())

    rules = learn_rules(
        train=train,
        label_col=args.label_col,
        categorical_columns=args.categorical_columns,
        null_columns=args.null_columns,
        min_support=args.min_support,
        min_purity=args.min_purity,
    )
    if not rules:
        raise RuntimeError("没有学到高置信规则，请降低 --min-support 或 --min-purity")
    print(f"\n从训练集学到 {len(rules)} 条规则：")
    for i, rule in enumerate(rules, start=1):
        value = "NULL" if rule.value is None else repr(rule.value)
        print(
            f"  {i}. {rule.column} {rule.kind} {value} -> {rule.pred_label}; "
            f"support={rule.support}, purity={rule.purity:.6f}"
        )

    pred, hits = predict(test, rules)
    result = pd.DataFrame(
        {
            "event_id": test[args.id_col].astype("string"),
            "pred_label": pred,
        }
    )
    if len(result) != len(test):
        raise AssertionError("预测行数与测试集不一致")
    if result["event_id"].isna().any() or result["event_id"].duplicated().any():
        raise AssertionError("输出 event_id 不完整或重复")
    if not set(result["pred_label"].unique()).issubset(LABELS):
        raise AssertionError("输出包含非法标签")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    args.rules_output.parent.mkdir(parents=True, exist_ok=True)
    args.rules_output.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "default_label": "benign",
                "min_support": args.min_support,
                "min_purity": args.min_purity,
                "rules": [asdict(r) for r in rules],
                "test_rule_hits": hits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n预测结果: {args.output.resolve()} ({len(result):,} 行)")
    print("预测分布:", result["pred_label"].value_counts().to_dict())
    print(f"规则文件: {args.rules_output.resolve()}")

    if not args.no_eval and answer_path.exists():
        evaluate(result, answer_path, args.id_col, args.label_col)
    elif not args.no_eval:
        print(f"未找到答案文件，跳过本地评分: {answer_path}")
    print(f"总耗时: {time.time() - started:.1f} 秒")


if __name__ == "__main__":
    main()
