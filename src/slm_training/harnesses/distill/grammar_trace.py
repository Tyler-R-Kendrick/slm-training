"""Grammar-state-conditioned decision traces for CAP1-02.

These traces capture the legal action distribution, selected/gold actions,
margins, entropy, and optional sensitivity at individual decode decisions.
They are estimated evidence: legal membership always comes from the compiler
owner and never from the teacher or model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

SCHEMA_VERSION = "cap1-02.v1"
_COVERAGE_VALUES = frozenset({"complete", "partial", "none"})
_CONVENTIONS = frozenset({"logit", "energy"})


@dataclass(frozen=True)
class GrammarDecisionTrace:
    """One decision point in a grammar-constrained decode rollout."""

    run_id: str
    checkpoint_id: str
    dataset_id: str
    example_id: str
    seed: int
    state_fingerprint: str
    state_signature_version: str
    decision_index: int
    diffusion_timestep: int | None
    legal_action_ids: tuple[str, ...]
    legal_action_kinds: tuple[str, ...]
    compiler_coverage: str
    selected_action_id: str | None
    target_action_ids: tuple[str, ...]
    target_semantics: str
    logits_or_energies: tuple[float, ...] | None
    normalized_probs: tuple[float, ...] | None
    top1_margin: float | None
    posterior_entropy_bits: float | None
    scope_signature: str
    expected_type: str | None
    template_signature: str | None
    completion_support_size_exact: int | None
    model_feature_ref: str | None
    sensitivity: Mapping[str, float] | None
    verification_outcome: str | None
    trace_schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema_version": self.trace_schema_version,
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "dataset_id": self.dataset_id,
            "example_id": self.example_id,
            "seed": self.seed,
            "state_fingerprint": self.state_fingerprint,
            "state_signature_version": self.state_signature_version,
            "decision_index": self.decision_index,
            "diffusion_timestep": self.diffusion_timestep,
            "legal_action_ids": list(self.legal_action_ids),
            "legal_action_kinds": list(self.legal_action_kinds),
            "compiler_coverage": self.compiler_coverage,
            "selected_action_id": self.selected_action_id,
            "target_action_ids": list(self.target_action_ids),
            "target_semantics": self.target_semantics,
            "logits_or_energies": (
                list(self.logits_or_energies)
                if self.logits_or_energies is not None
                else None
            ),
            "normalized_probs": (
                list(self.normalized_probs) if self.normalized_probs is not None else None
            ),
            "top1_margin": self.top1_margin,
            "posterior_entropy_bits": self.posterior_entropy_bits,
            "scope_signature": self.scope_signature,
            "expected_type": self.expected_type,
            "template_signature": self.template_signature,
            "completion_support_size_exact": self.completion_support_size_exact,
            "model_feature_ref": self.model_feature_ref,
            "sensitivity": dict(self.sensitivity) if self.sensitivity is not None else None,
            "verification_outcome": self.verification_outcome,
        }


class GrammarTraceRecorder:
    """Collect GrammarDecisionTrace rows during decode.

    Attach to a model as ``grammar_trace_recorder``. The decode path emits
    records only when the recorder is present, so default decode is unchanged.
    """

    def __init__(
        self,
        *,
        run_id: str = "",
        checkpoint_id: str = "",
        dataset_id: str = "",
        example_id: str = "",
        seed: int = 0,
        capture_logits: bool = False,
        capture_sensitivity: str | None = None,
        max_sensitivity_records: int | None = None,
        state_stratified: bool = False,
    ) -> None:
        self.run_id = run_id
        self.checkpoint_id = checkpoint_id
        self.dataset_id = dataset_id
        self.example_id = example_id
        self.seed = seed
        self.capture_logits = capture_logits
        self.capture_sensitivity = capture_sensitivity
        self.max_sensitivity_records = max_sensitivity_records
        self.state_stratified = state_stratified
        self._records: list[GrammarDecisionTrace] = []
        self._decision_index = 0
        self._sensitivity_count = 0
        self._seen_states: set[str] = set()

    def record(
        self,
        *,
        state_fingerprint: str,
        state_signature_version: str = "1",
        legal_action_ids: Sequence[str],
        legal_action_kinds: Sequence[str] | None = None,
        compiler_coverage: str = "partial",
        selected_action_id: str | None = None,
        target_action_ids: Sequence[str] | None = None,
        target_semantics: str = "",
        logits_or_energies: Sequence[float] | None = None,
        convention: Literal["logit", "energy"] = "logit",
        diffusion_timestep: int | None = None,
        scope_signature: str = "",
        expected_type: str | None = None,
        template_signature: str | None = None,
        completion_support_size_exact: int | None = None,
        model_feature_ref: str | None = None,
        sensitivity: Mapping[str, float] | None = None,
        verification_outcome: str | None = None,
    ) -> GrammarDecisionTrace | None:
        """Record one grammar-state decision.

        Returns the recorded trace or None when state-stratified sampling skips
        a duplicate state.
        """
        if self.state_stratified and state_fingerprint in self._seen_states:
            return None
        self._seen_states.add(state_fingerprint)

        legal_action_ids = tuple(legal_action_ids)
        legal_action_kinds = (
            tuple(legal_action_kinds) if legal_action_kinds is not None else ()
        )
        target_action_ids = tuple(target_action_ids or ())

        normalized_probs: tuple[float, ...] | None = None
        margin: float | None = None
        entropy: float | None = None

        if logits_or_energies is not None and self.capture_logits:
            values = tuple(float(x) for x in logits_or_energies)
            normalized_probs = normalize_legal_probs(values, convention=convention)
            margin = compute_margin(
                values,
                selected_index=_index_of(selected_action_id, legal_action_ids),
                convention=convention,
            )
            entropy = compute_entropy(normalized_probs)
            logits_or_energies = values
        else:
            logits_or_energies = None

        use_sensitivity: Mapping[str, float] | None = None
        if (
            sensitivity is not None
            and self.capture_sensitivity
            and (
                self.max_sensitivity_records is None
                or self._sensitivity_count < self.max_sensitivity_records
            )
        ):
            use_sensitivity = dict(sensitivity)
            self._sensitivity_count += 1

        self._decision_index += 1
        trace = GrammarDecisionTrace(
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            dataset_id=self.dataset_id,
            example_id=self.example_id,
            seed=self.seed,
            state_fingerprint=state_fingerprint,
            state_signature_version=state_signature_version,
            decision_index=self._decision_index,
            diffusion_timestep=diffusion_timestep,
            legal_action_ids=legal_action_ids,
            legal_action_kinds=legal_action_kinds,
            compiler_coverage=compiler_coverage,
            selected_action_id=selected_action_id,
            target_action_ids=target_action_ids,
            target_semantics=target_semantics,
            logits_or_energies=logits_or_energies,
            normalized_probs=normalized_probs,
            top1_margin=margin,
            posterior_entropy_bits=entropy,
            scope_signature=scope_signature,
            expected_type=expected_type,
            template_signature=template_signature,
            completion_support_size_exact=completion_support_size_exact,
            model_feature_ref=model_feature_ref,
            sensitivity=use_sensitivity,
            verification_outcome=verification_outcome,
        )
        self._records.append(trace)
        return trace

    def finalize(self) -> list[dict[str, Any]]:
        """Return all recorded traces as plain dicts."""
        return [record.to_dict() for record in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[GrammarDecisionTrace]:
        return iter(self._records)


def normalize_legal_probs(
    values: Sequence[float],
    *,
    convention: Literal["logit", "energy"] = "logit",
) -> tuple[float, ...]:
    """Softmax over the legal action set.

    For energies (lower is better) we negate before exponentiating.
    """
    if convention not in _CONVENTIONS:
        raise ValueError(f"convention must be one of {_CONVENTIONS}")
    if not values:
        return ()
    if convention == "energy":
        shifted = [-float(v) for v in values]
    else:
        shifted = [float(v) - max(values) for v in values]
    exps = [math.exp(v) for v in shifted]
    total = sum(exps)
    if total == 0.0:
        n = len(values)
        return tuple(1.0 / n for _ in range(n))
    return tuple(e / total for e in exps)


def compute_entropy(probs: Sequence[float]) -> float | None:
    """Shannon entropy in bits over the given distribution."""
    if not probs:
        return None
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log2(p)
    return entropy


def compute_margin(
    values: Sequence[float],
    *,
    selected_index: int | None = None,
    convention: Literal["logit", "energy"] = "logit",
) -> float | None:
    """Margin between best accepted and best competing legal action.

    For logits (higher better) margin = best_accepted - best_competing.
    For energies (lower better) margin = best_competing - best_accepted.
    When ``selected_index`` is provided and non-negative, it is treated as the
    single accepted action; otherwise the best-scoring action is accepted.
    """
    if not values or convention not in _CONVENTIONS:
        return None
    if convention == "logit":
        best = max(values)
        accepted = values[selected_index] if selected_index is not None and selected_index >= 0 else best
        competing = max(
            (v for i, v in enumerate(values) if i != selected_index),
            default=best,
        )
        return accepted - competing
    # energy: lower is better
    best = min(values)
    accepted = values[selected_index] if selected_index is not None and selected_index >= 0 else best
    competing = min(
        (v for i, v in enumerate(values) if i != selected_index),
        default=best,
    )
    return competing - accepted


def state_fingerprint(
    *,
    prefix_ids: Sequence[int],
    legal_action_ids: Sequence[str],
    coverage: str,
    signature_version: str = "1",
) -> str:
    """Deterministic fingerprint of a grammar decision state."""
    payload = {
        "signature_version": signature_version,
        "prefix_ids": list(prefix_ids),
        "legal_action_ids": sorted(legal_action_ids),
        "coverage": coverage,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def legal_action_ids_from_state(
    tokenizer: Any,
    state: Any,
    prefix_ids: Sequence[int],
) -> tuple[list[str], str] | None:
    """Return canonical legal action ids and coverage from a GrammarDecodeState.

    Falls back to the fastpath engine if the state exposes one. Returns None
    when no engine is available.
    """
    engine = getattr(state, "engine", None)
    if engine is None:
        return None
    try:
        from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            build_completion_forest,
        )

        terminals = engine.next_terminals()
        allowed = allowed_id_set(tokenizer, terminals)
        forest = build_completion_forest(
            tokenizer,
            list(prefix_ids),
            state=state,
        )
        candidate_ids = set(forest.candidate_ids) if forest.candidate_ids else allowed
        legal = sorted(
            {tokenizer.id_to_token.get(tid, str(tid)) for tid in candidate_ids}
        )
        return legal, str(forest.coverage)
    except Exception:  # noqa: BLE001
        return None


def grammar_trace_replay_violations(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Lightweight replay checks for a sequence of grammar-decision records."""
    violations: list[str] = []
    for idx, record in enumerate(records):
        prefix = f"record {idx}"
        coverage = record.get("compiler_coverage")
        if coverage not in _COVERAGE_VALUES:
            violations.append(f"{prefix}: invalid compiler_coverage {coverage!r}")
        legal = record.get("legal_action_ids") or []
        selected = record.get("selected_action_id")
        if selected is not None and selected not in legal:
            violations.append(
                f"{prefix}: selected_action_id {selected!r} not in legal_action_ids"
            )
        entropy = record.get("posterior_entropy_bits")
        if entropy is not None:
            n = len(legal)
            if n == 0:
                violations.append(f"{prefix}: entropy defined but no legal actions")
            elif not 0.0 <= entropy <= math.log2(max(n, 2)):
                violations.append(
                    f"{prefix}: entropy {entropy} out of [0, log2({n})]"
                )
        probs = record.get("normalized_probs") or []
        if probs and abs(sum(probs) - 1.0) > 1e-5:
            violations.append(f"{prefix}: normalized_probs sum to {sum(probs)}, not 1.0")
    return violations


def grammar_trace_coverage_report(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate estimated-evidence report over grammar-decision traces."""
    if not records:
        return {"n": 0, "note": "no records"}

    states: set[str] = set()
    state_action_pairs: set[tuple[str, str]] = set()
    scopes: set[str] = set()
    templates: set[str] = set()
    branching: list[int] = []
    margins: list[float] = []
    entropies: list[float] = []
    partial_records = 0
    missing_scores = 0
    checkpoints: set[str] = set()
    datasets: set[str] = set()
    seeds: set[int] = set()

    for record in records:
        state = record.get("state_fingerprint", "")
        states.add(state)
        legal = record.get("legal_action_ids") or []
        selected = record.get("selected_action_id")
        branching.append(len(legal))
        for action in legal:
            state_action_pairs.add((state, action))
        if selected is not None:
            state_action_pairs.add((state, selected))
        scopes.add(record.get("scope_signature") or "")
        templates.add(record.get("template_signature") or "")
        if record.get("compiler_coverage") != "complete":
            partial_records += 1
        if record.get("top1_margin") is None or record.get("posterior_entropy_bits") is None:
            missing_scores += 1
        margin = record.get("top1_margin")
        if margin is not None:
            margins.append(margin)
        entropy = record.get("posterior_entropy_bits")
        if entropy is not None:
            entropies.append(entropy)
        checkpoints.add(record.get("checkpoint_id", ""))
        datasets.add(record.get("dataset_id", ""))
        seeds.add(record.get("seed", 0))

    def _hist(values: Sequence[float], bins: int = 5) -> dict[str, int]:
        if not values:
            return {}
        lo, hi = min(values), max(values)
        if hi == lo:
            return {"single": len(values)}
        width = (hi - lo) / bins
        counts: dict[str, int] = {}
        for v in values:
            bin_idx = min(int((v - lo) / width), bins - 1)
            key = f"{lo + bin_idx * width:.4g}-{(lo + (bin_idx + 1) * width):.4g}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    return {
        "n": len(records),
        "unique_states": len(states),
        "state_action_pairs": len(state_action_pairs),
        "scope_signatures": len(scopes),
        "template_signatures": len(templates),
        "branching": {
            "min": min(branching) if branching else None,
            "max": max(branching) if branching else None,
            "mean": sum(branching) / len(branching) if branching else None,
        },
        "margin_histogram": _hist(margins),
        "entropy_histogram": _hist(entropies),
        "partial_coverage_records": partial_records,
        "missing_score_records": missing_scores,
        "forced_decision_fraction": (
            sum(1 for b in branching if b == 1) / len(branching) if branching else None
        ),
        "provenance": {
            "checkpoints": sorted(checkpoints),
            "datasets": sorted(datasets),
            "seeds": sorted(seeds),
        },
    }


def _index_of(value: str | None, values: Sequence[str]) -> int | None:
    if value is None:
        return None
    try:
        return values.index(value)
    except ValueError:
        return None
