from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soc_threat.content_model import ContentThreatModel  # noqa: E402


class ContentModelTests(unittest.TestCase):
    def test_content_only_forward_shape_and_neutral_padding(self) -> None:
        model = ContentThreatModel(
            mode="content",
            hash_buckets=128,
            content_embedding_dim=8,
            content_output_dim=16,
            cardinalities=[],
            numeric_count=0,
            class_count=3,
            token_dropout=0.0,
            category_dropout=0.0,
        )
        tokens = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]], dtype=torch.long)
        categorical = torch.empty((2, 0), dtype=torch.long)
        numeric = torch.empty((2, 0), dtype=torch.float32)
        fused, content = model(categorical, numeric, tokens)
        self.assertEqual(tuple(fused.shape), (2, 3))
        self.assertTrue(torch.equal(fused, content))
        self.assertTrue(
            torch.equal(
                model.content_encoder.embedding.weight[0],
                torch.zeros(8),
            )
        )

    def test_fusion_forward_shape(self) -> None:
        model = ContentThreatModel(
            mode="fusion_field",
            hash_buckets=128,
            content_embedding_dim=8,
            content_output_dim=16,
            cardinalities=[4, 5],
            numeric_count=3,
            class_count=3,
            token_dropout=0.0,
            category_dropout=0.0,
        )
        model.eval()
        tokens = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
        categorical = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
        numeric = torch.ones((2, 3), dtype=torch.float32)
        fused, content = model(categorical, numeric, tokens)
        self.assertEqual(tuple(fused.shape), (2, 3))
        self.assertEqual(tuple(content.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
