"""Default-off rate model over exact dynamic legal-edit candidate batches."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.flow.targets import LegalEditRateTargetV1
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_scorer import (
    DIRECT_POLICY_SCHEMA,
    LegalEditScorer,
    LegalEditScorerConfig,
)

FLOW_POLICY_SCHEMA = "legal_edit_flow_policy/v1"


@dataclass(frozen=True)
class LegalEditFlowConfig:
    """Flow is opt-in so existing training and decode remain unchanged."""

    enabled: bool = False
    min_rate: float = 1e-8
    scorer: LegalEditScorerConfig = LegalEditScorerConfig(time_encoding="linear")

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_rate) or self.min_rate <= 0.0:
            raise ValueError("min_rate must be positive and finite")


@dataclass(frozen=True)
class FlowPrediction:
    edge_rates: torch.Tensor
    row_hazards: torch.Tensor
    terminal_logits: torch.Tensor


class LegalEditFlow(nn.Module):
    """Turn the shared objective-neutral scorer into nonnegative edge rates."""

    def __init__(self, config: LegalEditFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalEditFlowConfig()
        self.scorer = LegalEditScorer(self.config.scorer)
        self.terminal_head = nn.Linear(2, 1)

    def forward(
        self,
        batch: LegalEditBatch,
        *,
        schedule_progress: torch.Tensor | None = None,
    ) -> FlowPrediction:
        if not self.config.enabled:
            raise RuntimeError("legal-edit flow is disabled; opt in explicitly")
        logits = self.scorer(batch, schedule_progress=schedule_progress)
        rates = F.softplus(logits) + self.config.min_rate
        hazards = rates.new_zeros(len(batch.row_ids))
        hazards.index_add_(0, batch.candidate_to_row, rates)
        terminal_logits = self.terminal_head(batch.state_features[:, :2]).squeeze(-1)
        return FlowPrediction(rates, hazards, terminal_logits)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        torch.save(
            {
                "schema": FLOW_POLICY_SCHEMA,
                "config": {
                    "enabled": self.config.enabled,
                    "min_rate": self.config.min_rate,
                    "scorer": asdict(self.config.scorer),
                },
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LegalEditFlow":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") == DIRECT_POLICY_SCHEMA:
            # Migration is deliberately default-off. It copies the shared
            # scorer while leaving the new terminal head inert until opt-in.
            scorer_config = LegalEditScorerConfig(**dict(payload.get("config") or {}))
            model = cls(LegalEditFlowConfig(enabled=False, scorer=scorer_config))
            model.scorer.load_state_dict(payload["state_dict"], strict=True)
            return model
        if payload.get("schema") != FLOW_POLICY_SCHEMA:
            raise ValueError("legal-edit-flow checkpoint schema mismatch")
        raw = dict(payload.get("config") or {})
        raw["scorer"] = LegalEditScorerConfig(**dict(raw.get("scorer") or {}))
        model = cls(LegalEditFlowConfig(**raw))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class ExactRateTable(nn.Module):
    """Tiny exact-fixture seam: one learnable log-rate per finite graph edge."""

    def __init__(self, targets: tuple[LegalEditRateTargetV1, ...]) -> None:
        super().__init__()
        self.row_ids = tuple(target.row_id for target in targets)
        self.candidate_ids = tuple(target.candidate_ids for target in targets)
        self.log_rates = nn.ParameterList(
            [nn.Parameter(torch.zeros(len(target.candidate_ids))) for target in targets]
        )

    def forward(self) -> tuple[torch.Tensor, ...]:
        return tuple(F.softplus(value) for value in self.log_rates)


def legal_edit_flow_losses(
    prediction: FlowPrediction,
    batch: LegalEditBatch,
    targets: tuple[LegalEditRateTargetV1, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return separate edge, hazard, set-mass, and terminal objectives."""
    if len(targets) != len(batch.row_ids):
        raise ValueError("target count does not match batch rows")
    target_rates = prediction.edge_rates.new_zeros(len(batch.candidate_ids))
    target_hazards = prediction.row_hazards.new_zeros(len(targets))
    terminal = prediction.terminal_logits.new_zeros(len(targets))
    supervised = torch.zeros(
        len(batch.candidate_ids),
        dtype=torch.bool,
        device=prediction.edge_rates.device,
    )
    hazard_supervised = torch.zeros(
        len(targets), dtype=torch.bool, device=prediction.row_hazards.device
    )
    set_losses: list[torch.Tensor] = []
    for row, target in enumerate(targets):
        start, end = int(batch.row_offsets[row]), int(batch.row_offsets[row + 1])
        if target.row_id != batch.row_ids[row]:
            raise ValueError("target row identity differs from the exact batch")
        ids = batch.candidate_ids[start:end]
        if ids != target.candidate_ids:
            raise ValueError("target membership/order differs from the exact batch")
        target_rates[start:end] = target_rates.new_tensor(target.edge_rates)
        supervised[start:end] = torch.tensor(
            [item in target.supervised_candidate_ids for item in ids],
            dtype=torch.bool,
            device=supervised.device,
        )
        target_hazards[row] = target.total_hazard
        hazard_supervised[row] = target.hazard_supervised
        terminal[row] = target.terminal_probability
        positive = target_rates[start:end] > 0
        probabilities = prediction.edge_rates[start:end] / prediction.edge_rates[
            start:end
        ].sum()
        if bool(positive.any()):
            set_losses.append(
                -torch.log(probabilities[positive].sum().clamp_min(1e-12))
            )
    edge_rate_loss = (
        F.mse_loss(prediction.edge_rates[supervised], target_rates[supervised])
        if bool(supervised.any())
        else prediction.edge_rates.sum() * 0.0
    )
    losses = {
        "edge_rate": edge_rate_loss,
        "total_hazard": (
            F.mse_loss(
                prediction.row_hazards[hazard_supervised],
                target_hazards[hazard_supervised],
            )
            if bool(hazard_supervised.any())
            else prediction.row_hazards.sum() * 0.0
        ),
        "multi_positive_mass": (
            torch.stack(set_losses).mean()
            if set_losses
            else prediction.edge_rates.sum() * 0.0
        ),
        "terminal_absorption": F.binary_cross_entropy_with_logits(
            prediction.terminal_logits, terminal
        ),
    }
    return sum(losses.values()), losses
