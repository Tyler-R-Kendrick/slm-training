"""Frozen constraint-debt telemetry rows for preference decision events."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import torch
import torch.nn.functional as F

from slm_training.harnesses.preference.local_decisions import DecisionEventV1

ProbabilitySpace = Literal["full_vocab", "legal_tokens"]


def _sha256_stable_identity(event: DecisionEventV1) -> str:
    """Stable state id fallback when ``event_id`` is empty."""
    import hashlib
    import json

    payload = {
        "group_id": event.group_id,
        "trajectory_id": event.trajectory_id,
        "policy_checkpoint_sha": event.policy_checkpoint_sha,
        "tokenizer_sha": event.tokenizer_sha,
        "decode_config_hash": event.decode_config_hash,
        "decision_kind": event.decision_kind,
        "split": event.split,
        "canvas_ids": list(event.canvas_ids),
        "position": event.position,
        "legal_token_ids": list(event.legal_token_ids),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ConstraintDebtV1:
    """Frozen telemetry row describing grammar-mask probability distortion.

    This is pure instrumentation: it never changes the objective, decoder, or
    gradient flow.  When ``probability_space == "legal_tokens"`` the masses are
    over the verifier-legal renormalized distribution; otherwise they are over
    the full vocabulary softmax.
    """

    schema_version: int = 1
    state_id: str
    group_id: str
    trajectory_id: str | None
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    verifier_bundle_hash: str
    decision_kind: str
    abstract_state_role: str | None
    split: str
    probability_space: ProbabilitySpace
    epsilon: float
    full_vocab_log_normalizer: float
    legal_mass: float
    good_mass: float
    bad_mass: float
    ambiguous_mass: float
    unobserved_mass: float
    legal_debt: float | None
    good_debt: float | None
    bad_debt: float | None
    legal_mass_deficit: float
    pre_post_mask_kl: float
    legal_support_count: int
    good_support_count: int
    bad_support_count: int
    single_legal_action: bool
    empty_good_partition: bool
    empty_bad_partition: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConstraintDebtV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown constraint debt fields: {sorted(unknown)}")
        return cls(**value)


@torch.no_grad()
def _constraint_debt_mass_bundle(
    logits: torch.Tensor, event: DecisionEventV1
) -> dict[str, torch.Tensor | dict[int, int]]:
    """Return detached full-vocab and legal-renormalized probability tensors.

    The returned dictionaries contain:

    * ``full_probs`` — softmax over the full vocabulary.
    * ``legal_probs`` — softmax over ``event.legal_token_ids``.
    * ``legal_index`` — token id -> position in ``legal_probs``.
    * ``legal_ids`` — tensor of legal token ids.
    """
    if logits.ndim != 1:
        raise ValueError("constraint debt logits must be one-dimensional")
    legal_ids = torch.tensor(
        event.legal_token_ids, dtype=torch.long, device=logits.device
    )
    full_probs = F.softmax(logits, dim=-1)
    legal_logits = logits.index_select(0, legal_ids)
    legal_probs = F.softmax(legal_logits, dim=-1)
    legal_index = {
        token_id: index for index, token_id in enumerate(event.legal_token_ids)
    }
    return {
        "full_probs": full_probs,
        "legal_probs": legal_probs,
        "legal_index": legal_index,
        "legal_ids": legal_ids,
    }


def _debt(mass: float, epsilon: float) -> float:
    """Return ``-log(mass + epsilon)`` for a non-empty partition."""
    return -math.log(mass + epsilon)


@torch.no_grad()
def compute_constraint_debt_v1(
    logits: torch.Tensor,
    event: DecisionEventV1,
    *,
    probability_space: ProbabilitySpace = "full_vocab",
    epsilon: float = 1e-12,
) -> ConstraintDebtV1:
    """Compute a frozen :class:`ConstraintDebtV1` row for ``event``.

    The computation is wrapped in ``torch.no_grad()`` so it can be attached to
    training/diagnostic paths without affecting gradients.
    """
    if probability_space not in ("full_vocab", "legal_tokens"):
        raise ValueError(f"unknown probability space: {probability_space}")

    bundle = _constraint_debt_mass_bundle(logits, event)
    full_probs = bundle["full_probs"]
    legal_probs = bundle["legal_probs"]
    legal_index = bundle["legal_index"]
    legal_ids_tensor = bundle["legal_ids"]

    good_ids = event.good_token_ids
    bad_ids = event.bad_token_ids
    legal_ids = event.legal_token_ids

    full_vocab_log_normalizer = float(torch.logsumexp(logits, dim=0))

    if probability_space == "full_vocab":
        legal_mass = float(full_probs.index_select(0, legal_ids_tensor).sum())
        good_mass = float(
            full_probs.index_select(
                0,
                torch.tensor(good_ids, dtype=torch.long, device=logits.device),
            ).sum()
        )
        bad_mass = float(
            full_probs.index_select(
                0,
                torch.tensor(bad_ids, dtype=torch.long, device=logits.device),
            ).sum()
        )
        legal_mass_deficit = 1.0 - legal_mass
        unobserved_mass = 1.0 - legal_mass
    else:
        legal_mass = 1.0
        legal_good_ids = torch.tensor(
            tuple(legal_index[token_id] for token_id in good_ids),
            dtype=torch.long,
            device=logits.device,
        )
        legal_bad_ids = torch.tensor(
            tuple(legal_index[token_id] for token_id in bad_ids),
            dtype=torch.long,
            device=logits.device,
        )
        good_mass = float(legal_probs.index_select(0, legal_good_ids).sum())
        bad_mass = float(legal_probs.index_select(0, legal_bad_ids).sum())
        legal_mass_deficit = 0.0
        unobserved_mass = 0.0

    ambiguous_mass = max(0.0, legal_mass - good_mass - bad_mass)

    legal_debt = _debt(legal_mass, epsilon) if legal_ids else None
    good_debt = _debt(good_mass, epsilon) if good_ids else None
    bad_debt = _debt(bad_mass, epsilon) if bad_ids else None

    empty_good_partition = not good_ids
    empty_bad_partition = not bad_ids

    # Pre/post-mask KL: sum_{a in legal} legal_probs[a] * log(legal_probs[a] /
    # (full_probs[a] / legal_mass)).  Add epsilon to the denominator for
    # numerical stability when the full distribution places near-zero mass on a
    # legal token.
    full_legal_probs = full_probs.index_select(0, legal_ids_tensor)
    denominator = (full_legal_probs / max(legal_mass, epsilon)) + epsilon
    pre_post_mask_kl = float(
        (legal_probs * ((legal_probs + epsilon).log() - denominator.log())).sum()
    )

    return ConstraintDebtV1(
        state_id=event.event_id or _sha256_stable_identity(event),
        group_id=event.group_id,
        trajectory_id=event.trajectory_id,
        policy_checkpoint_sha=event.policy_checkpoint_sha,
        tokenizer_sha=event.tokenizer_sha,
        decode_config_hash=event.decode_config_hash,
        verifier_bundle_hash="",
        decision_kind=event.decision_kind,
        abstract_state_role=None,
        split=event.split,
        probability_space=probability_space,
        epsilon=epsilon,
        full_vocab_log_normalizer=full_vocab_log_normalizer,
        legal_mass=legal_mass,
        good_mass=good_mass,
        bad_mass=bad_mass,
        ambiguous_mass=ambiguous_mass,
        unobserved_mass=unobserved_mass,
        legal_debt=legal_debt,
        good_debt=good_debt,
        bad_debt=bad_debt,
        legal_mass_deficit=legal_mass_deficit,
        pre_post_mask_kl=pre_post_mask_kl,
        legal_support_count=len(legal_ids),
        good_support_count=len(good_ids),
        bad_support_count=len(bad_ids),
        single_legal_action=len(legal_ids) == 1,
        empty_good_partition=empty_good_partition,
        empty_bad_partition=empty_bad_partition,
    )
