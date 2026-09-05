from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from soc_threat.content_model import (
    ContentEncoder,
    MultiViewContentEncoder,
    embedding_dimension,
)


NoveltyGateMode = Literal["none", "count"]
ContentInputMode = Literal["raw", "multiview"]


@dataclass(frozen=True)
class HierarchicalOutput:
    threat_logits: torch.Tensor
    subtype_logits: torch.Tensor
    metadata_threat_logits: torch.Tensor
    content_threat_logits: torch.Tensor
    metadata_subtype_logits: torch.Tensor
    applied_gate: torch.Tensor


class CategoricalNumericEncoder(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        numeric_count: int,
        *,
        hidden_dim: int,
        output_dim: int,
        category_dropout: float,
    ) -> None:
        super().__init__()
        self.category_dropout = category_dropout
        dimensions = [embedding_dimension(value) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension, padding_idx=0)
                for cardinality, dimension in zip(cardinalities, dimensions)
            ]
        )
        input_dim = sum(dimensions) + numeric_count
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.category_dropout > 0:
            drop = (
                torch.rand(categorical.shape, device=categorical.device)
                < self.category_dropout
            )
            categorical = categorical.masked_fill(drop, 0)
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat([*embedded, numeric], dim=1))


class HierarchicalContentModel(nn.Module):
    def __init__(
        self,
        *,
        novelty_gate_mode: NoveltyGateMode,
        hash_buckets: int,
        content_embedding_dim: int,
        content_output_dim: int,
        metadata_cardinalities: list[int],
        metadata_numeric_count: int,
        semantic_cardinalities: list[int],
        semantic_numeric_count: int,
        token_dropout: float,
        category_dropout: float,
        content_input_mode: ContentInputMode = "raw",
        content_view_count: int = 4,
        content_tokens_per_view: int = 64,
    ) -> None:
        super().__init__()
        if novelty_gate_mode not in {"none", "count"}:
            raise ValueError(f"Unknown novelty gate mode: {novelty_gate_mode}")
        if content_input_mode not in {"raw", "multiview"}:
            raise ValueError(f"Unknown content input mode: {content_input_mode}")
        self.novelty_gate_mode = novelty_gate_mode
        self.content_input_mode = content_input_mode
        self.metadata_encoder = CategoricalNumericEncoder(
            metadata_cardinalities,
            metadata_numeric_count,
            hidden_dim=256,
            output_dim=128,
            category_dropout=category_dropout,
        )
        self.semantic_encoder = CategoricalNumericEncoder(
            semantic_cardinalities,
            semantic_numeric_count,
            hidden_dim=96,
            output_dim=64,
            category_dropout=category_dropout,
        )
        if content_input_mode == "raw":
            self.content_encoder = ContentEncoder(
                hash_buckets=hash_buckets,
                embedding_dim=content_embedding_dim,
                output_dim=content_output_dim,
                token_dropout=token_dropout,
            )
        else:
            self.content_encoder = MultiViewContentEncoder(
                hash_buckets=hash_buckets,
                embedding_dim=content_embedding_dim,
                output_dim=content_output_dim,
                token_dropout=token_dropout,
                view_count=content_view_count,
                tokens_per_view=content_tokens_per_view,
            )

        combined_dim = 128 + 64 + content_output_dim
        self.threat_head = nn.Sequential(
            nn.Linear(combined_dim, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(160, 2),
        )
        self.metadata_threat_head = nn.Linear(128, 2)
        self.content_threat_head = nn.Linear(content_output_dim, 2)

        self.metadata_subtype_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.subtype_residual_head = nn.Sequential(
            nn.Linear(64 + content_output_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        # Start from the metadata-only subtype classifier. Semantic content is
        # allowed to change it only after the residual has learned evidence.
        residual_output = self.subtype_residual_head[-1]
        if not isinstance(residual_output, nn.Linear):
            raise TypeError("Subtype residual output must be linear")
        nn.init.zeros_(residual_output.weight)
        nn.init.zeros_(residual_output.bias)

    def forward(
        self,
        metadata_categorical: torch.Tensor,
        metadata_numeric: torch.Tensor,
        semantic_categorical: torch.Tensor,
        semantic_numeric: torch.Tensor,
        token_ids: torch.Tensor,
        novelty_gate: torch.Tensor,
    ) -> HierarchicalOutput:
        metadata = self.metadata_encoder(metadata_categorical, metadata_numeric)
        semantic = self.semantic_encoder(semantic_categorical, semantic_numeric)
        content = self.content_encoder(token_ids)

        threat_logits = self.threat_head(
            torch.cat([metadata, semantic, content], dim=1)
        )
        metadata_threat_logits = self.metadata_threat_head(metadata)
        content_threat_logits = self.content_threat_head(content)

        metadata_subtype_logits = self.metadata_subtype_head(metadata)
        subtype_residual = self.subtype_residual_head(
            torch.cat([semantic, content], dim=1)
        )
        if self.novelty_gate_mode == "none":
            applied_gate = torch.ones_like(novelty_gate)
        else:
            applied_gate = novelty_gate.clamp(0.0, 1.0)
        subtype_logits = (
            metadata_subtype_logits + applied_gate.unsqueeze(1) * subtype_residual
        )
        return HierarchicalOutput(
            threat_logits=threat_logits,
            subtype_logits=subtype_logits,
            metadata_threat_logits=metadata_threat_logits,
            content_threat_logits=content_threat_logits,
            metadata_subtype_logits=metadata_subtype_logits,
            applied_gate=applied_gate,
        )
