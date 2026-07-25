"""Objective-neutral scoring and direct policies over exact legal-edit sets."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.data.flow.bridge_corpus import (
    RequestEditContractV1,
    canonical_fingerprint,
    enumerate_live_candidates,
    parse_statements,
    verify_certificate,
)
from slm_training.flow.termination import (
    ABSTAIN,
    HOLD,
    NO_LIVE_CANDIDATES,
    FixedKPolicy,
    TerminationContext,
    TerminationPolicy,
)
from slm_training.lineage.records import content_sha
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    CanonicalEdit,
    apply_canonical_edit,
)
from slm_training.models.legal_edit_batch import FEATURE_NAMES, LegalEditBatch

DIRECT_POLICY_SCHEMA = "direct_legal_edit_policy/v1"
TimeEncoding = Literal["no_time", "linear", "fourier"]


@dataclass(frozen=True)
class LegalEditScorerConfig:
    hidden_dim: int = 32
    time_encoding: TimeEncoding = "no_time"
    plan_enabled: bool = False
    seed: int = 0
    scorer_id: str = "legal-edit-scorer-v1"

    def __post_init__(self) -> None:
        if self.time_encoding not in {"no_time", "linear", "fourier"}:
            raise ValueError(f"unknown time encoding: {self.time_encoding}")


@dataclass(frozen=True)
class DirectPolicyDecision:
    candidate_id: str | None
    decision_kind: Literal["scored", "forced", "abstain"]
    log_probability: float
    model_calls: int
    reason: str = ""


@dataclass(frozen=True)
class DirectDecodeTrace:
    final_program: str
    final_fingerprint: str
    stop_reason: str
    decisions: tuple[dict[str, Any], ...]
    model_calls: int
    elapsed_seconds: float


class LegalEditScorer(nn.Module):
    """Shared scorer; the objective controls labels, never candidate membership."""

    def __init__(self, config: LegalEditScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalEditScorerConfig()
        torch.manual_seed(self.config.seed)
        hidden = self.config.hidden_dim
        self.state_projection = nn.Linear(2, hidden)
        self.candidate_projection = nn.Linear(len(FEATURE_NAMES), hidden)
        self.time_projection = nn.Linear(2, hidden)
        self.plan_projection = nn.Linear(1, hidden)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _time_features(
        self, progress: torch.Tensor, *, encoding: TimeEncoding | None = None
    ) -> torch.Tensor:
        mode = encoding or self.config.time_encoding
        if mode == "no_time":
            return torch.zeros((progress.numel(), 2), device=progress.device)
        if mode == "linear":
            return torch.stack((progress, 1.0 - progress), dim=-1)
        if mode == "fourier":
            return torch.stack(
                (torch.sin(math.pi * progress), torch.cos(math.pi * progress)),
                dim=-1,
            )
        raise ValueError(f"unknown time encoding: {mode}")

    def forward(
        self,
        batch: LegalEditBatch,
        *,
        schedule_progress: torch.Tensor | None = None,
        plan_hint: torch.Tensor | None = None,
        time_encoding: TimeEncoding | None = None,
    ) -> torch.Tensor:
        row_count = len(batch.row_ids)
        progress = (
            schedule_progress
            if schedule_progress is not None
            else torch.zeros(row_count, device=batch.state_features.device)
        )
        if progress.shape != (row_count,) or bool(((progress < 0) | (progress > 1)).any()):
            raise ValueError("schedule_progress must have shape [B] and lie in [0, 1]")
        if plan_hint is None:
            plan_hint = torch.zeros(
                len(batch.candidate_ids), device=batch.candidate_features.device
            )
        if plan_hint.shape != (len(batch.candidate_ids),):
            raise ValueError("plan_hint must have exact shape [N]")

        # Gold-derived normalized_progress/sample_time never enter the scorer.
        safe_state = batch.state_features[:, :2]
        state = self.state_projection(safe_state)[batch.candidate_to_row]
        candidates = self.candidate_projection(batch.candidate_features)
        encoded_time = self.time_projection(self._time_features(progress, encoding=time_encoding))
        encoded_time = encoded_time[batch.candidate_to_row]
        encoded_plan = self.plan_projection(plan_hint[:, None])
        return self.score_head(
            torch.cat((state, candidates, encoded_time, encoded_plan), dim=-1)
        ).squeeze(-1)

    def artifact_identity(self) -> dict[str, Any]:
        return {
            "schema": DIRECT_POLICY_SCHEMA,
            "config": asdict(self.config),
            "param_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def multi_positive_set_loss(
    logits: torch.Tensor, batch: LegalEditBatch
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return -log probability mass assigned to every certified positive."""
    if logits.shape != (len(batch.candidate_ids),):
        raise ValueError("logits must have exact shape [N]")
    losses: list[torch.Tensor] = []
    positive_mass: list[float] = []
    unknown_mass: list[float] = []
    for row in range(len(batch.row_ids)):
        start, end = int(batch.row_offsets[row]), int(batch.row_offsets[row + 1])
        positives = batch.positive_mask[start:end]
        if not bool(positives.any()):
            raise ValueError("each training row requires a certified positive")
        row_logits = logits[start:end]
        losses.append(
            torch.logsumexp(row_logits, dim=0)
            - torch.logsumexp(row_logits[positives], dim=0)
        )
        probabilities = F.softmax(row_logits, dim=0)
        positive_mass.append(float(probabilities[positives].sum().detach()))
        unknown_mass.append(
            float(probabilities[batch.unknown_mask[start:end]].sum().detach())
        )
    return torch.stack(losses).mean(), {
        "positive_mass": sum(positive_mass) / len(positive_mass),
        "unknown_mass": sum(unknown_mass) / len(unknown_mass),
        "rows": float(len(losses)),
    }


class DirectLegalEditPolicy:
    """Greedy/stochastic direct policy over compiler-enumerated candidates."""

    def __init__(self, scorer: LegalEditScorer) -> None:
        self.scorer = scorer

    def decide(
        self,
        batch: LegalEditBatch,
        *,
        schedule_progress: torch.Tensor | None = None,
        plan_hint: torch.Tensor | None = None,
        stochastic: bool = False,
        generator: torch.Generator | None = None,
    ) -> DirectPolicyDecision:
        if len(batch.row_ids) != 1:
            raise ValueError("direct decode accepts exactly one state")
        count = len(batch.candidate_ids)
        if count == 0:
            return DirectPolicyDecision(None, "abstain", 0.0, 0, NO_LIVE_CANDIDATES)
        if count == 1:
            return DirectPolicyDecision(batch.candidate_ids[0], "forced", 0.0, 0)
        logits = self.scorer(
            batch, schedule_progress=schedule_progress, plan_hint=plan_hint
        )
        probabilities = F.softmax(logits, dim=0)
        index = (
            int(torch.multinomial(probabilities, 1, generator=generator).item())
            if stochastic
            else int(probabilities.argmax().item())
        )
        return DirectPolicyDecision(
            batch.candidate_ids[index],
            "scored",
            float(probabilities[index].log().detach()),
            1,
        )

    def decode_exact(
        self,
        source: str,
        contract: RequestEditContractV1,
        *,
        termination: TerminationPolicy | None = None,
        stochastic: bool = False,
        seed: int = 0,
        max_steps: int = 4,
        max_wall_seconds: float = 30.0,
    ) -> DirectDecodeTrace:
        termination = termination or FixedKPolicy(k=max_steps, max_steps=max_steps)
        generator = torch.Generator().manual_seed(seed)
        current = source.strip()
        started = time.monotonic()
        trace: list[dict[str, Any]] = []
        model_calls = 0
        stop_reason = ABSTAIN
        for step in range(max_steps + 1):
            elapsed = time.monotonic() - started
            candidate_set = enumerate_live_candidates(current, contract)
            stop = termination.decide(
                TerminationContext(
                    state_fingerprint=canonical_fingerprint(current),
                    step_index=step,
                    edit_count=step,
                    wall_time=elapsed,
                    candidates=tuple(item.candidate_id for item in candidate_set.candidates),
                )
            )
            if stop.action != HOLD:
                stop_reason = stop.reason
                break
            if elapsed > max_wall_seconds:
                stop_reason = "wall_budget"
                break
            statements = parse_statements(current) or ()
            batch = LegalEditBatch.pack_inference(
                candidate_set, statement_count=len(statements), step_index=step
            )
            decision = self.decide(
                batch,
                schedule_progress=torch.tensor([step / max(1, max_steps)]),
                stochastic=stochastic,
                generator=generator,
            )
            model_calls += decision.model_calls
            if decision.candidate_id is None:
                stop_reason = decision.reason
                break
            candidate = next(
                item
                for item in candidate_set.candidates
                if item.candidate_id == decision.candidate_id
            )
            verify_certificate(candidate.transition_certificate)
            successor = apply_canonical_edit(
                current, CanonicalEdit.from_dict(candidate.edit)
            )
            if (
                successor is None
                or canonical_fingerprint(successor) != candidate.successor_fingerprint
            ):
                stop_reason = "invalid_replay"
                break
            trace.append(
                {
                    "step": step,
                    "state_fingerprint": candidate_set.state_fingerprint,
                    "candidate_set_digest": candidate_set.candidate_set_digest,
                    "candidate_ids": list(batch.candidate_ids),
                    "selected_candidate_id": decision.candidate_id,
                    "decision_kind": decision.decision_kind,
                    "log_probability": decision.log_probability,
                    "successor_fingerprint": candidate.successor_fingerprint,
                }
            )
            current = successor
        return DirectDecodeTrace(
            final_program=current,
            final_fingerprint=canonical_fingerprint(current),
            stop_reason=stop_reason,
            decisions=tuple(trace),
            model_calls=model_calls,
            elapsed_seconds=time.monotonic() - started,
        )

    def save(
        self, path: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> None:
        payload = {
            "schema": DIRECT_POLICY_SCHEMA,
            "config": asdict(self.scorer.config),
            "state_dict": self.scorer.state_dict(),
            "artifact_identity": self.scorer.artifact_identity(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "DirectLegalEditPolicy":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != DIRECT_POLICY_SCHEMA:
            raise ValueError("direct-policy checkpoint schema mismatch")
        config = dict(payload.get("config") or {})
        config.setdefault("time_encoding", "no_time")
        scorer = LegalEditScorer(LegalEditScorerConfig(**config))
        scorer.load_state_dict(payload["state_dict"], strict=True)
        return cls(scorer)

    @staticmethod
    def checkpoint_sha(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def compatibility_fingerprint(self) -> str:
        return content_sha(self.scorer.artifact_identity())
