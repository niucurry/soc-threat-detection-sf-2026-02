from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.anchored_residual_model import (  # noqa: E402
    AnchoredConflictResidualModel,
)
from soc_threat.hierarchical_model import HierarchicalContentModel  # noqa: E402


def make_anchor() -> HierarchicalContentModel:
    return HierarchicalContentModel(
        novelty_gate_mode="count",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        metadata_cardinalities=[4, 5],
        metadata_numeric_count=3,
        semantic_cardinalities=[4, 5, 3, 6],
        semantic_numeric_count=6,
        token_dropout=0.0,
        category_dropout=0.0,
    ).eval()


def inputs(batch_size: int = 2):
    return (
        torch.tensor([[1, 2]] * batch_size),
        torch.randn(batch_size, 3),
        torch.tensor([[1, 2, 1, 3]] * batch_size),
        torch.randn(batch_size, 6),
        torch.tensor([[2, 3, 4, 0]] * batch_size),
        None,
        torch.full((batch_size,), 0.75),
        torch.full((batch_size,), 100.0),
    )


def test_zero_initialized_residual_exactly_reproduces_anchor() -> None:
    anchor = make_anchor()
    model = AnchoredConflictResidualModel(
        anchor=anchor,
        residual_input_mode="anchor",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        token_dropout=0.0,
        residual_hidden_dim=16,
    ).eval()
    values = inputs()
    output = model(*values)
    direct = anchor(
        values[0], values[1], values[2], values[3], values[4], values[6]
    )
    torch.testing.assert_close(output.threat_logits, direct.threat_logits)
    torch.testing.assert_close(output.subtype_logits, direct.subtype_logits)
    torch.testing.assert_close(output.delta_margin, torch.zeros(2))
    assert all(not parameter.requires_grad for parameter in model.anchor.parameters())


def test_residual_changes_only_final_benign_metadata_threat_conflicts() -> None:
    anchor = make_anchor()
    final_output = anchor.threat_head[-1]
    assert isinstance(final_output, nn.Linear)
    with torch.no_grad():
        final_output.weight.zero_()
        final_output.bias.copy_(torch.tensor([2.0, -2.0]))
        anchor.metadata_threat_head.weight.zero_()
        anchor.metadata_threat_head.bias.copy_(torch.tensor([-2.0, 2.0]))
    model = AnchoredConflictResidualModel(
        anchor=anchor,
        residual_input_mode="anchor",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        token_dropout=0.0,
        residual_hidden_dim=16,
    ).eval()
    trust_output = model.trust_head[-1]
    assert isinstance(trust_output, nn.Linear)
    with torch.no_grad():
        trust_output.bias.fill_(10.0)
    output = model(*inputs())
    assert output.conflict_mask.tolist() == [True, True]
    assert torch.all(output.delta_margin > 0)
    assert torch.all(output.threat_logits[:, 1] > output.anchor_threat_logits[:, 1])

    with torch.no_grad():
        anchor.metadata_threat_head.bias.copy_(torch.tensor([2.0, -2.0]))
    no_conflict = model(*inputs())
    assert no_conflict.conflict_mask.tolist() == [False, False]
    torch.testing.assert_close(no_conflict.threat_logits, no_conflict.anchor_threat_logits)


def test_content_evidence_rescues_only_content_threat_conflicts() -> None:
    anchor = make_anchor()
    final_output = anchor.threat_head[-1]
    assert isinstance(final_output, nn.Linear)
    with torch.no_grad():
        final_output.weight.zero_()
        final_output.bias.copy_(torch.tensor([2.0, -2.0]))
        anchor.metadata_threat_head.weight.zero_()
        anchor.metadata_threat_head.bias.copy_(torch.tensor([2.0, -2.0]))
        anchor.content_threat_head.weight.zero_()
        anchor.content_threat_head.bias.copy_(torch.tensor([-2.0, 2.0]))
    model = AnchoredConflictResidualModel(
        anchor=anchor,
        residual_input_mode="anchor",
        evidence_source="content",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        token_dropout=0.0,
        residual_hidden_dim=16,
    ).eval()
    trust_output = model.trust_head[-1]
    assert isinstance(trust_output, nn.Linear)
    with torch.no_grad():
        trust_output.bias.fill_(10.0)
    output = model(*inputs())
    assert output.metadata_candidate.tolist() == [False, False]
    assert output.content_candidate.tolist() == [True, True]
    assert output.evidence_candidate.tolist() == [True, True]
    assert output.conflict_mask.tolist() == [True, True]
    assert torch.all(output.delta_margin > 0)

    with torch.no_grad():
        anchor.content_threat_head.bias.copy_(torch.tensor([2.0, -2.0]))
    no_conflict = model(*inputs())
    assert no_conflict.content_candidate.tolist() == [False, False]
    assert no_conflict.conflict_mask.tolist() == [False, False]
    torch.testing.assert_close(no_conflict.threat_logits, no_conflict.anchor_threat_logits)


def test_multiview_encoder_starts_from_v7_token_encoder() -> None:
    anchor = make_anchor()
    model = AnchoredConflictResidualModel(
        anchor=anchor,
        residual_input_mode="multiview",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        token_dropout=0.0,
        content_view_count=4,
        content_tokens_per_view=8,
        residual_hidden_dim=16,
    )
    assert model.multiview_encoder is not None
    for expected, actual in zip(
        anchor.content_encoder.parameters(),
        model.multiview_encoder.shared_encoder.parameters(),
    ):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() != expected.data_ptr()


def test_train_mode_keeps_anchor_in_eval_mode() -> None:
    model = AnchoredConflictResidualModel(
        anchor=make_anchor(),
        residual_input_mode="anchor",
        hash_buckets=128,
        content_embedding_dim=8,
        content_output_dim=16,
        token_dropout=0.0,
        residual_hidden_dim=16,
    )
    model.train()
    assert model.training
    assert not model.anchor.training
    assert model.trust_head.training
