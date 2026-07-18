"""CAP2-03 state-local action head fixture harness.

Runs all five state-local action-head families on deterministic fixture states
with known legal action sets.  The oracle pass wires the correct codeword or
factor logits directly, proving each family can recover the legal action.  The
random-init pass confirms the head path runs to completion and only returns
legal actions (or abstains) without training.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from slm_training.models.action_code_registry import ActionCodeRegistry
from slm_training.models.local_action_head import (
    GlobalMaskedHead,
    GrammarFactorizedHead,
    LocalActionHead,
    LocalFlatHead,
    StateContext,
    TernaryDigitHead,
    TernaryECOCHead,
)
from slm_training.models.semantic_cost import (
    ordinal_base3_codeword,
)


HIDDEN_DIM = 16


@dataclass(frozen=True)
class FixtureState:
    """One fixture compiler state."""

    state_family_id: str
    state_signature: tuple[str, ...]
    legal_actions: tuple[str, ...]
    correct_action: str


@dataclass(frozen=True)
class HeadResult:
    """Result for one head family on the fixture set."""

    head_family: str
    oracle_accuracy: float
    random_init_accuracy: float
    forced_states_recovered: int
    abstain_count: int
    detected_error_count: int
    elapsed_seconds: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_family": self.head_family,
            "oracle_accuracy": self.oracle_accuracy,
            "random_init_accuracy": self.random_init_accuracy,
            "forced_states_recovered": self.forced_states_recovered,
            "abstain_count": self.abstain_count,
            "detected_error_count": self.detected_error_count,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StateLocalActionReport:
    """Versioned fixture report for CAP2-03."""

    run_id: str
    version: str
    timestamp: str
    hidden_dim: int
    states: tuple[FixtureState, ...]
    results: tuple[HeadResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "hidden_dim": self.hidden_dim,
            "states": [
                {
                    "state_family_id": s.state_family_id,
                    "state_signature": s.state_signature,
                    "legal_actions": s.legal_actions,
                    "correct_action": s.correct_action,
                }
                for s in self.states
            ],
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def fixture_states() -> tuple[FixtureState, ...]:
    """Deterministic fixture states with known legal action sets."""
    return (
        FixtureState(
            "card_root",
            ("root",),
            (
                "component:root:none:card",
                "component:root:none:text",
                "component:root:none:button",
                "literal:root:none:none",
            ),
            "component:root:none:card",
        ),
        FixtureState(
            "bind_arg0",
            ("arg0",),
            (
                "bind:arg0:local:text",
                "bind:arg0:global:card",
                "literal:arg0:none:none",
            ),
            "bind:arg0:local:text",
        ),
        FixtureState(
            "forced_leaf",
            ("leaf",),
            ("literal:leaf:none:none",),
            "literal:leaf:none:none",
        ),
    )


def _make_head(head_family: str) -> LocalActionHead:
    if head_family == "global_masked":
        return GlobalMaskedHead(HIDDEN_DIM, max_vocabulary=32)
    if head_family == "local_flat":
        return LocalFlatHead(HIDDEN_DIM)
    if head_family == "ternary_digit":
        return TernaryDigitHead(HIDDEN_DIM)
    if head_family == "ternary_ecoc":
        return TernaryECOCHead(HIDDEN_DIM, registry=ActionCodeRegistry())
    if head_family == "grammar_factorized":
        return GrammarFactorizedHead(HIDDEN_DIM)
    raise ValueError(f"unknown head family {head_family!r}")


def _oracle_output(
    head: LocalActionHead,
    state: FixtureState,
    device: torch.device,
) -> Any:
    """Build an output tensor that encodes the correct action for ``head``."""
    legal = list(state.legal_actions)
    correct = state.correct_action
    batch = 1

    if isinstance(head, GlobalMaskedHead):
        logits = torch.full((batch, head.max_vocabulary), float("-inf"), device=device)
        correct_idx = hash(correct) % head.max_vocabulary
        logits[0, correct_idx] = 10.0
        out = head.score(torch.zeros(batch, HIDDEN_DIM, device=device), StateContext(""), legal)
        out.logits = logits
        return out

    if isinstance(head, LocalFlatHead):
        out = head.score(torch.zeros(batch, HIDDEN_DIM, device=device), StateContext(""), legal)
        assert out.logits is not None
        scores = torch.full_like(out.logits, -10.0)
        correct_pos = legal.index(correct)
        scores[0, correct_pos] = 10.0
        out.logits = scores
        return out

    if isinstance(head, TernaryDigitHead):
        b = len(legal)
        m = math.ceil(math.log(b, 3)) if b > 1 else 0
        cw = ordinal_base3_codeword(legal.index(correct), m)
        trits = torch.full((batch, m, 3), -10.0, device=device)
        for pos, trit in enumerate(cw):
            trits[0, pos, trit] = 10.0
        out = head.score(torch.zeros(batch, HIDDEN_DIM, device=device), StateContext(""), legal)
        out.trits = trits
        return out

    if isinstance(head, TernaryECOCHead):
        entry = head._get_entry(legal)
        cw = entry.codeword_for(correct)
        assert cw is not None
        m = len(cw)
        trits = torch.full((batch, m, 3), -10.0, device=device)
        for pos, trit in enumerate(cw):
            trits[0, pos, trit] = 10.0
        out = head.score(torch.zeros(batch, HIDDEN_DIM, device=device), StateContext(""), legal)
        out.trits = trits
        return out

    if isinstance(head, GrammarFactorizedHead):
        out = head.score(torch.zeros(batch, HIDDEN_DIM, device=device), StateContext(""), legal)
        factors = head._parse_action(correct)
        for name, logits in out.factor_logits.items():
            biased = torch.full_like(logits, -10.0)
            biased[0, factors[name]] = 10.0
            out.factor_logits[name] = biased
        return out

    raise ValueError(f"unsupported head family {head.head_family}")


def _evaluate_head(head_family: str, states: tuple[FixtureState, ...]) -> HeadResult:
    start = time.monotonic()
    device = torch.device("cpu")
    head = _make_head(head_family).to(device)
    notes: list[str] = []

    oracle_correct = 0
    random_legal = 0
    forced_recovered = 0
    abstain_count = 0
    detected_error_count = 0

    for state in states:
        legal = list(state.legal_actions)
        ctx = StateContext(state_family_id=state.state_family_id)

        # Oracle pass: wiring the correct code/factors must recover the action.
        oracle_out = _oracle_output(head, state, device)
        oracle_decision = head.decode(oracle_out, legal)
        if oracle_decision.action_identity == state.correct_action:
            oracle_correct += 1

        # Random-init pass: no training, just verify the path completes legally.
        hidden = torch.randn(1, HIDDEN_DIM, device=device)
        random_out = head.score(hidden, ctx, legal)
        random_decision = head.decode(random_out, legal)
        if random_decision.decision_kind == "forced":
            forced_recovered += 1
            if random_decision.action_identity == state.correct_action:
                random_legal += 1
        elif random_decision.action_identity in legal:
            random_legal += 1
        elif random_decision.decision_kind == "abstain":
            abstain_count += 1
        elif random_decision.decision_kind == "detected_error":
            detected_error_count += 1

    oracle_accuracy = oracle_correct / len(states)
    random_init_accuracy = random_legal / len(states)

    if oracle_accuracy < 1.0:
        notes.append("oracle pass did not recover every correct action")
    else:
        notes.append("oracle pass recovered every correct action")
    notes.append(
        f"random-init pass: {random_init_accuracy:.2%} legal/forced, "
        f"{abstain_count} abstain, {detected_error_count} detected_error"
    )

    return HeadResult(
        head_family=head_family,
        oracle_accuracy=oracle_accuracy,
        random_init_accuracy=random_init_accuracy,
        forced_states_recovered=forced_recovered,
        abstain_count=abstain_count,
        detected_error_count=detected_error_count,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def run_fixture(
    head_families: tuple[str, ...] | None = None,
) -> StateLocalActionReport:
    """Run the CAP2-03 state-local action-head fixture matrix."""
    if head_families is None:
        head_families = (
            "global_masked",
            "local_flat",
            "ternary_digit",
            "ternary_ecoc",
            "grammar_factorized",
        )
    states = fixture_states()
    results = tuple(_evaluate_head(family, states) for family in head_families)
    run_id = _hash_run_id(("cap2-03", head_families, [s.state_family_id for s in states]))
    return StateLocalActionReport(
        run_id=run_id,
        version="cap2-03-v1",
        timestamp=_utc_now(),
        hidden_dim=HIDDEN_DIM,
        states=states,
        results=results,
    )
