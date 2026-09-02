from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.hierarchical_features import (  # noqa: E402
    HIERARCHICAL_REQUIRED_COLUMNS,
    fit_hierarchical_preprocessor,
    transform_hierarchical_inputs,
)


def _row(
    *,
    product: str,
    family: str,
    action: str,
    label: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: 0 for column in HIERARCHICAL_REQUIRED_COLUMNS}
    for column in HIERARCHICAL_REQUIRED_COLUMNS:
        if column in {
            "pipeline",
            "product_name",
            "product_group",
            "src_ip_kind",
            "port_bucket",
            "message_length_bucket",
            "structure_combo",
            "network_missing_pattern",
            "vendor_name",
            "content_family",
            "content_action",
            "content_protocol",
            "content_event_code",
        }:
            row[column] = "__MISSING__"
    row.update(
        {
            "pipeline": "syslog",
            "product_name": product,
            "product_group": "missing" if product == "__MISSING__" else "aws_vpc",
            "vendor_name": (
                "__MISSING__" if product == "__MISSING__" else "Amazon Web Services"
            ),
            "content_family": family,
            "content_action": action,
            "content_protocol": "tcp",
            "content_event_code": "__MISSING__",
            "raw_token_count": 96,
            "label_binary": label,
        }
    )
    return row


def test_novelty_gate_is_zero_for_unseen_product_semantic_combo() -> None:
    train = pd.DataFrame(
        [
            _row(
                product="AWS VPC Security",
                family="vpc_flow",
                action="reject",
                label="suspicious",
            ),
            _row(
                product="AWS VPC Security",
                family="vpc_flow",
                action="reject",
                label="suspicious",
            ),
        ]
    )
    valid = pd.DataFrame(
        [
            _row(
                product="__MISSING__",
                family="vpc_flow",
                action="reject",
                label="malicious",
            )
        ]
    )
    preprocessor = fit_hierarchical_preprocessor(train, novelty_pseudocount=2.0)
    train_arrays = transform_hierarchical_inputs(train, preprocessor)
    valid_arrays = transform_hierarchical_inputs(valid, preprocessor)

    np.testing.assert_array_equal(train_arrays.combo_counts, [2, 2])
    np.testing.assert_allclose(train_arrays.novelty_gate, [0.5, 0.5])
    np.testing.assert_array_equal(valid_arrays.combo_counts, [0])
    np.testing.assert_array_equal(valid_arrays.novelty_gate, [0.0])


def test_combination_frequency_does_not_depend_on_labels() -> None:
    rows = [
        _row(
            product="AWS VPC Security",
            family="vpc_flow",
            action="reject",
            label="suspicious",
        ),
        _row(
            product="__MISSING__",
            family="syslog_text",
            action="deny",
            label="malicious",
        ),
    ]
    original = pd.DataFrame(rows)
    changed = original.copy()
    changed["label_binary"] = ["benign", "benign"]

    first = fit_hierarchical_preprocessor(original)
    second = fit_hierarchical_preprocessor(changed)
    assert first["combo_counts"] == second["combo_counts"]
