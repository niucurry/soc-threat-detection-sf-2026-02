from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.content_features import hash_symbol  # noqa: E402
from soc_threat.hierarchical_model import HierarchicalContentModel  # noqa: E402
from soc_threat.multiview_content_features import encode_multiview_content  # noqa: E402
from train_hierarchical_content import (  # noqa: E402
    positive_evidence_preservation_loss,
)


def test_middle_view_retains_a_signal_outside_head_and_tail() -> None:
    message = "alpha " * 80 + "middlethreat " + "omega " * 80
    encoded = encode_multiview_content(
        message,
        buckets=65_536,
        tokens_per_view=32,
        chars_per_view=256,
    )
    head = encoded.multiview_token_ids[:32]
    middle = encoded.multiview_token_ids[32:64]
    tail = encoded.multiview_token_ids[64:96]
    marker = hash_symbol("w:middlethreat")
    assert marker not in head
    assert marker in middle
    assert marker not in tail


def test_key_value_view_preserves_field_value_relationship() -> None:
    message = 'payload={"result":"invalid_passcode","source":"10.1.2.3"}'
    encoded = encode_multiview_content(
        message,
        buckets=65_536,
        tokens_per_view=64,
    )
    key_value = encoded.multiview_token_ids[192:256]
    assert hash_symbol("kv_key:result") in key_value
    assert hash_symbol("kv_pair:result=invalid_passcode") in key_value


def test_multiview_network_accepts_flattened_four_view_tokens() -> None:
    model = HierarchicalContentModel(
        novelty_gate_mode="count",
        content_input_mode="multiview",
        content_view_count=4,
        content_tokens_per_view=32,
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
    output = model(
        torch.tensor([[1, 2], [2, 3]]),
        torch.randn(2, 3),
        torch.tensor([[1, 2, 1, 3], [2, 3, 1, 4]]),
        torch.randn(2, 6),
        torch.randint(0, 128, (2, 128)),
        torch.tensor([0.0, 0.8]),
    )
    assert output.threat_logits.shape == (2, 2)
    assert output.content_threat_logits.shape == (2, 2)


def test_evidence_preservation_penalizes_positive_branch_suppression() -> None:
    final = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    metadata = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    content = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    targets = torch.tensor([1, 1])
    loss = positive_evidence_preservation_loss(
        final,
        metadata,
        content,
        targets,
        positive_margin=0.0,
        allowed_branch_gap=0.5,
    )
    assert float(loss) > 3.0

    no_positive = positive_evidence_preservation_loss(
        final,
        metadata,
        content,
        torch.zeros(2, dtype=torch.long),
        positive_margin=0.0,
        allowed_branch_gap=0.5,
    )
    assert float(no_positive) == 0.0
