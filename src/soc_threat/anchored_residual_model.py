from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from soc_threat.content_model import MultiViewContentEncoder
from soc_threat.hierarchical_model import HierarchicalContentModel


ResidualInputMode = Literal["anchor", "multiview"]
EvidenceSource = Literal["metadata", "content"]


@dataclass(frozen=True)
class FrozenAnchorOutput:
    metadata: torch.Tensor
    semantic: torch.Tensor
    raw_content: torch.Tensor
    threat_logits: torch.Tensor
    subtype_logits: torch.Tensor
    metadata_threat_logits: torch.Tensor
    content_threat_logits: torch.Tensor
    metadata_subtype_logits: torch.Tensor
    applied_gate: torch.Tensor
    metadata_candidate: torch.Tensor
    content_candidate: torch.Tensor
    evidence_candidate: torch.Tensor
    conflict_mask: torch.Tensor


@dataclass(frozen=True)
class AnchoredResidualOutput:
    threat_logits: torch.Tensor
    subtype_logits: torch.Tensor
    anchor_threat_logits: torch.Tensor
    metadata_threat_logits: torch.Tensor
    content_threat_logits: torch.Tensor
    metadata_subtype_logits: torch.Tensor
    applied_gate: torch.Tensor
    metadata_candidate: torch.Tensor
    content_candidate: torch.Tensor
    evidence_candidate: torch.Tensor
    conflict_mask: torch.Tensor
    trust_logit: torch.Tensor
    trust_score: torch.Tensor
    delta_margin: torch.Tensor


class AnchoredConflictResidualModel(nn.Module):
    """A frozen v4.0 anchor with a narrowly scoped, exactly-zero residual.

    The residual is allowed to act only when the frozen final threat head says
    benign while one configured evidence branch says threat. Its maximum
    positive correction is the detached logit gap between those two branches.
    v5.0 uses metadata evidence; v5.1/v5.2 use content evidence. At
    initialization the residual output is exactly zero, so predictions and
    probabilities are exactly those of the anchor.
    """

    def __init__(
        self,
        *,
        anchor: HierarchicalContentModel,
        residual_input_mode: ResidualInputMode,
        evidence_source: EvidenceSource = "metadata",
        hash_buckets: int,
        content_embedding_dim: int,
        content_output_dim: int,
        token_dropout: float,
        content_view_count: int = 4,
        content_tokens_per_view: int = 64,
        residual_hidden_dim: int = 128,
        max_conflict_gap: float = 24.0,
    ) -> None:
        super().__init__()
        if residual_input_mode not in {"anchor", "multiview"}:
            raise ValueError(f"Unknown residual input mode: {residual_input_mode}")
        if evidence_source not in {"metadata", "content"}:
            raise ValueError(f"Unknown evidence source: {evidence_source}")
        if anchor.content_input_mode != "raw":
            raise ValueError("The frozen v4.0 anchor must use raw-token content")
        if max_conflict_gap <= 0:
            raise ValueError("max_conflict_gap must be positive")

        self.anchor = anchor
        self.residual_input_mode = residual_input_mode
        self.evidence_source = evidence_source
        self.max_conflict_gap = float(max_conflict_gap)
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()

        self.multiview_encoder: MultiViewContentEncoder | None
        if residual_input_mode == "multiview":
            self.multiview_encoder = MultiViewContentEncoder(
                hash_buckets=hash_buckets,
                embedding_dim=content_embedding_dim,
                output_dim=content_output_dim,
                token_dropout=token_dropout,
                view_count=content_view_count,
                tokens_per_view=content_tokens_per_view,
            )
            # Initialize from v4.0 token semantics. Both this copied encoder and
            # the view-fusion layer remain trainable; the anchor itself is frozen.
            self.multiview_encoder.shared_encoder.load_state_dict(
                self.anchor.content_encoder.state_dict()
            )
        else:
            self.multiview_encoder = None

        # Frozen v4.0 vectors: metadata=128, semantic=64, raw content=output_dim.
        # Four scalars describe metadata/content confidence and novelty.  The
        # final anchor margin is deliberately excluded from the trust input:
        # otherwise an in-sample trust classifier can merely copy the anchor instead
        # of learning when the configured evidence branch is reliable.
        input_dim = 128 + 64 + content_output_dim + 4
        if self.multiview_encoder is not None:
            input_dim += content_output_dim
        self.trust_head = nn.Sequential(
            nn.Linear(input_dim, residual_hidden_dim),
            nn.LayerNorm(residual_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(residual_hidden_dim, 1),
        )
        output = self.trust_head[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("Trust head output must be linear")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def train(self, mode: bool = True) -> AnchoredConflictResidualModel:
        super().train(mode)
        # Parent train() recursively changes every child.  The anchor must stay
        # in inference mode so dropout cannot move the supposedly fixed target.
        self.anchor.eval()
        return self

    def anchor_forward(
        self,
        metadata_categorical: torch.Tensor,
        metadata_numeric: torch.Tensor,
        semantic_categorical: torch.Tensor,
        semantic_numeric: torch.Tensor,
        raw_token_ids: torch.Tensor,
        novelty_gate: torch.Tensor,
    ) -> FrozenAnchorOutput:
        with torch.no_grad():
            metadata = self.anchor.metadata_encoder(
                metadata_categorical, metadata_numeric
            )
            semantic = self.anchor.semantic_encoder(
                semantic_categorical, semantic_numeric
            )
            raw_content = self.anchor.content_encoder(raw_token_ids)
            threat_logits = self.anchor.threat_head(
                torch.cat([metadata, semantic, raw_content], dim=1)
            )
            metadata_threat_logits = self.anchor.metadata_threat_head(metadata)
            content_threat_logits = self.anchor.content_threat_head(raw_content)
            metadata_subtype_logits = self.anchor.metadata_subtype_head(metadata)
            subtype_residual = self.anchor.subtype_residual_head(
                torch.cat([semantic, raw_content], dim=1)
            )
            if self.anchor.novelty_gate_mode == "none":
                applied_gate = torch.ones_like(novelty_gate)
            else:
                applied_gate = novelty_gate.clamp(0.0, 1.0)
            subtype_logits = (
                metadata_subtype_logits
                + applied_gate.unsqueeze(1) * subtype_residual
            )
            anchor_margin = threat_logits[:, 1] - threat_logits[:, 0]
            metadata_margin = (
                metadata_threat_logits[:, 1] - metadata_threat_logits[:, 0]
            )
            content_margin = (
                content_threat_logits[:, 1] - content_threat_logits[:, 0]
            )
            metadata_candidate = metadata_margin > 0
            content_candidate = content_margin > 0
            evidence_candidate = (
                metadata_candidate
                if self.evidence_source == "metadata"
                else content_candidate
            )
            conflict_mask = (anchor_margin < 0) & evidence_candidate
        return FrozenAnchorOutput(
            metadata=metadata,
            semantic=semantic,
            raw_content=raw_content,
            threat_logits=threat_logits,
            subtype_logits=subtype_logits,
            metadata_threat_logits=metadata_threat_logits,
            content_threat_logits=content_threat_logits,
            metadata_subtype_logits=metadata_subtype_logits,
            applied_gate=applied_gate,
            metadata_candidate=metadata_candidate,
            content_candidate=content_candidate,
            evidence_candidate=evidence_candidate,
            conflict_mask=conflict_mask,
        )

    def forward(
        self,
        metadata_categorical: torch.Tensor,
        metadata_numeric: torch.Tensor,
        semantic_categorical: torch.Tensor,
        semantic_numeric: torch.Tensor,
        raw_token_ids: torch.Tensor,
        multiview_token_ids: torch.Tensor | None,
        novelty_gate: torch.Tensor,
        combo_counts: torch.Tensor,
    ) -> AnchoredResidualOutput:
        frozen = self.anchor_forward(
            metadata_categorical,
            metadata_numeric,
            semantic_categorical,
            semantic_numeric,
            raw_token_ids,
            novelty_gate,
        )
        anchor_margin = frozen.threat_logits[:, 1] - frozen.threat_logits[:, 0]
        metadata_margin = (
            frozen.metadata_threat_logits[:, 1]
            - frozen.metadata_threat_logits[:, 0]
        )
        content_margin = (
            frozen.content_threat_logits[:, 1]
            - frozen.content_threat_logits[:, 0]
        )
        evidence_margin = (
            metadata_margin
            if self.evidence_source == "metadata"
            else content_margin
        )
        conflict_gap = (evidence_margin - anchor_margin).clamp(
            min=0.0, max=self.max_conflict_gap
        )
        scalar_context = torch.stack(
            [
                metadata_margin.clamp(-12.0, 12.0) / 12.0,
                content_margin.clamp(-12.0, 12.0) / 12.0,
                novelty_gate.clamp(0.0, 1.0),
                torch.log1p(combo_counts.to(anchor_margin.dtype)).clamp_max(12.0)
                / 12.0,
            ],
            dim=1,
        )
        context = [
            frozen.metadata.detach(),
            frozen.semantic.detach(),
            frozen.raw_content.detach(),
            scalar_context.detach(),
        ]
        if self.multiview_encoder is not None:
            if multiview_token_ids is None:
                raise ValueError("multiview_token_ids are required in multiview mode")
            context.append(self.multiview_encoder(multiview_token_ids))
        trust_logit = self.trust_head(torch.cat(context, dim=1)).squeeze(1)
        trust_score = torch.tanh(trust_logit).clamp_min(0.0)
        delta_margin = (
            frozen.conflict_mask.to(trust_score.dtype)
            * trust_score
            * conflict_gap.detach()
        )
        correction = torch.stack((-0.5 * delta_margin, 0.5 * delta_margin), dim=1)
        threat_logits = frozen.threat_logits + correction
        return AnchoredResidualOutput(
            threat_logits=threat_logits,
            subtype_logits=frozen.subtype_logits,
            anchor_threat_logits=frozen.threat_logits,
            metadata_threat_logits=frozen.metadata_threat_logits,
            content_threat_logits=frozen.content_threat_logits,
            metadata_subtype_logits=frozen.metadata_subtype_logits,
            applied_gate=frozen.applied_gate,
            metadata_candidate=frozen.metadata_candidate,
            content_candidate=frozen.content_candidate,
            evidence_candidate=frozen.evidence_candidate,
            conflict_mask=frozen.conflict_mask,
            trust_logit=trust_logit,
            trust_score=trust_score,
            delta_margin=delta_margin,
        )
