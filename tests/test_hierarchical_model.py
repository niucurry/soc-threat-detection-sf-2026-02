from __future__ import annotations

import numpy as np
import torch
from torch import nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.hierarchical_model import HierarchicalContentModel  # noqa: E402
from train_hierarchical_content import (  # noqa: E402
    hierarchical_predictions,
    hierarchical_probabilities,
)


def _model(mode: str) -> HierarchicalContentModel:
    return HierarchicalContentModel(
        novelty_gate_mode=mode,
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        metadata_cardinalities=[4, 5],
        metadata_numeric_count=3,
        semantic_cardinalities=[4, 5, 3, 6],
        semantic_numeric_count=6,
        token_dropout=0.0,
        category_dropout=0.0,
    )


def test_hierarchical_model_output_shapes() -> None:
    model = _model("count").eval()
    output = model(
        torch.tensor([[1, 2], [2, 3]]),
        torch.randn(2, 3),
        torch.tensor([[1, 2, 1, 3], [2, 3, 1, 4]]),
        torch.randn(2, 6),
        torch.tensor([[2, 3, 0], [4, 5, 6]]),
        torch.tensor([0.0, 0.8]),
    )
    assert output.threat_logits.shape == (2, 2)
    assert output.subtype_logits.shape == (2, 2)
    assert output.content_threat_logits.shape == (2, 2)
    assert output.applied_gate.tolist() == [0.0, 0.800000011920929]


def test_zero_novelty_gate_uses_metadata_subtype_only() -> None:
    model = _model("count").eval()
    residual_output = model.subtype_residual_head[-1]
    assert isinstance(residual_output, nn.Linear)
    with torch.no_grad():
        residual_output.bias.copy_(torch.tensor([2.0, -2.0]))
    common = (
        torch.tensor([[1, 2], [1, 2]]),
        torch.randn(2, 3),
        torch.tensor([[1, 2, 1, 3], [1, 2, 1, 3]]),
        torch.randn(2, 6),
        torch.tensor([[2, 3, 0], [2, 3, 0]]),
    )
    output = model(*common, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        output.subtype_logits[0], output.metadata_subtype_logits[0]
    )
    assert not torch.allclose(
        output.subtype_logits[1], output.metadata_subtype_logits[1]
    )


def test_hierarchical_probability_and_decision_are_consistent() -> None:
    threat = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]], dtype=np.float32)
    subtype = np.asarray([[0.5, 0.5], [0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    probabilities = hierarchical_probabilities(threat, subtype)
    predicted, threat_predicted, subtype_predicted = hierarchical_predictions(
        threat, subtype, threshold=0.5
    )

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(threat_predicted, [0, 1, 1])
    np.testing.assert_array_equal(subtype_predicted, [0, 0, 1])
    np.testing.assert_array_equal(predicted, [0, 1, 2])
