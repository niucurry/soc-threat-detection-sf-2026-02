from __future__ import annotations

from typing import Literal

import torch
from torch import nn


ModelMode = Literal["content", "fusion_raw", "fusion_field"]


def embedding_dimension(cardinality: int) -> int:
    return min(24, max(3, int(round(2.0 * cardinality**0.25))))


class ContentEncoder(nn.Module):
    def __init__(
        self,
        *,
        hash_buckets: int,
        embedding_dim: int,
        output_dim: int,
        token_dropout: float,
    ) -> None:
        super().__init__()
        self.token_dropout = token_dropout
        self.embedding = nn.Embedding(
            hash_buckets,
            embedding_dim,
            padding_idx=0,
        )
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.training and self.token_dropout > 0:
            keep = (
                torch.rand(token_ids.shape, device=token_ids.device)
                >= self.token_dropout
            )
            token_ids = token_ids.masked_fill((token_ids != 0) & ~keep, 0)
        mask = token_ids != 0
        embedded = self.embedding(token_ids)
        float_mask = mask.unsqueeze(-1).to(embedded.dtype)
        counts = float_mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (embedded * float_mask).sum(dim=1) / counts
        max_pool = embedded.masked_fill(~mask.unsqueeze(-1), -1.0e4).amax(dim=1)
        has_tokens = mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_tokens, max_pool, torch.zeros_like(max_pool))
        return self.network(torch.cat([mean_pool, max_pool], dim=1))


class MultiViewContentEncoder(nn.Module):
    """Encode fixed head/middle/tail/key-value views with shared token weights."""

    def __init__(
        self,
        *,
        hash_buckets: int,
        embedding_dim: int,
        output_dim: int,
        token_dropout: float,
        view_count: int,
        tokens_per_view: int,
    ) -> None:
        super().__init__()
        if view_count < 2 or tokens_per_view < 1:
            raise ValueError("Multi-view dimensions must be positive")
        self.view_count = view_count
        self.tokens_per_view = tokens_per_view
        self.shared_encoder = ContentEncoder(
            hash_buckets=hash_buckets,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            token_dropout=token_dropout,
        )
        self.view_embeddings = nn.Parameter(torch.zeros(view_count, output_dim))
        nn.init.normal_(self.view_embeddings, mean=0.0, std=0.02)
        self.fusion = nn.Sequential(
            nn.Linear(view_count * output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("Multi-view tokens must have shape [batch, flattened_tokens]")
        expected_width = self.view_count * self.tokens_per_view
        if token_ids.shape[1] != expected_width:
            raise ValueError(
                f"Expected {expected_width} multi-view tokens, got {token_ids.shape[1]}"
            )
        batch_size = token_ids.shape[0]
        views = token_ids.reshape(
            batch_size * self.view_count, self.tokens_per_view
        )
        encoded = self.shared_encoder(views).reshape(
            batch_size, self.view_count, -1
        )
        encoded = encoded + self.view_embeddings.unsqueeze(0)
        return self.fusion(encoded.flatten(start_dim=1))


class StructuredEncoder(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        numeric_count: int,
        *,
        output_dim: int,
        category_dropout: float,
    ) -> None:
        super().__init__()
        self.category_dropout = category_dropout
        dimensions = [embedding_dimension(value) for value in cardinalities]
        # Index zero is a deterministic neutral vector for validation categories
        # that were not observed while fitting the training preprocessor.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension, padding_idx=0)
                for cardinality, dimension in zip(cardinalities, dimensions)
            ]
        )
        input_size = sum(dimensions) + numeric_count
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
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


class ContentThreatModel(nn.Module):
    def __init__(
        self,
        *,
        mode: ModelMode,
        hash_buckets: int,
        content_embedding_dim: int,
        content_output_dim: int,
        cardinalities: list[int],
        numeric_count: int,
        class_count: int,
        token_dropout: float,
        category_dropout: float,
    ) -> None:
        super().__init__()
        if mode not in {"content", "fusion_raw", "fusion_field"}:
            raise ValueError(f"Unknown content model mode: {mode}")
        self.mode = mode
        self.content_encoder = ContentEncoder(
            hash_buckets=hash_buckets,
            embedding_dim=content_embedding_dim,
            output_dim=content_output_dim,
            token_dropout=token_dropout,
        )
        self.content_head = nn.Linear(content_output_dim, class_count)
        if mode == "content":
            self.structured_encoder = None
            self.fusion_head = None
        else:
            self.structured_encoder = StructuredEncoder(
                cardinalities,
                numeric_count,
                output_dim=128,
                category_dropout=category_dropout,
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(128 + content_output_dim, 192),
                nn.LayerNorm(192),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(192, 64),
                nn.SiLU(),
                nn.Linear(64, class_count),
            )

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        content_vector = self.content_encoder(token_ids)
        content_logits = self.content_head(content_vector)
        if self.mode == "content":
            return content_logits, content_logits
        if self.structured_encoder is None or self.fusion_head is None:
            raise RuntimeError("Fusion model was not initialized")
        structured_vector = self.structured_encoder(categorical, numeric)
        fused_logits = self.fusion_head(
            torch.cat([structured_vector, content_vector], dim=1)
        )
        return fused_logits, content_logits
