"""Grammar-stratified calibration corpus + low-bit adaptation for the local scorer.

This harness consumes CAP1-02 ``grammar_decision`` traces, builds a versioned
calibration corpus with deterministic stratified sampling, and runs small-scale
adaptation (PTQ scale calibration, QAT reconstruction, distillation objectives)
on a ``LocalActionHead`` target.  CPU/toy runs are wiring evidence only; they do
not claim ship-grade retention.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import torch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_training.harnesses.distill.grammar_trace import (
    grammar_trace_coverage_report,
    grammar_trace_replay_violations,
)
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.models.quantization.fake_quant import fake_quantize_weight
from slm_training.models.quantization.formats import QuantFormat

if TYPE_CHECKING:
    import torch

    from slm_training.models.local_action_head import LocalFlatHead


CALIBRATION_SCHEMA_VERSION = "cap3-02.v1"
PRIMARY_STRATEGIES = {
    "random_production",
    "uniform_state",
    "uniform_state_action",
    "scope_template_stratified",
    "low_margin",
    "sensitivity_weighted",
    "hybrid_coverage_margin",
}


@dataclass(frozen=True)
class CalibrationSample:
    """One row of the calibration corpus derived from a grammar decision."""

    trace_id: str
    state_fingerprint: str
    state_signature_version: str
    legal_action_ids: tuple[str, ...]
    selected_action_id: str | None
    target_action_ids: tuple[str, ...]
    top1_margin: float | None
    posterior_entropy_bits: float | None
    scope_signature: str
    template_signature: str | None
    production_weight: float
    bin_id: str | None
    sensitivity_score: float | None
    verification_outcome: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "state_fingerprint": self.state_fingerprint,
            "state_signature_version": self.state_signature_version,
            "legal_action_ids": list(self.legal_action_ids),
            "selected_action_id": self.selected_action_id,
            "target_action_ids": list(self.target_action_ids),
            "top1_margin": self.top1_margin,
            "posterior_entropy_bits": self.posterior_entropy_bits,
            "scope_signature": self.scope_signature,
            "template_signature": self.template_signature,
            "production_weight": self.production_weight,
            "bin_id": self.bin_id,
            "sensitivity_score": self.sensitivity_score,
            "verification_outcome": self.verification_outcome,
        }


@dataclass
class CalibrationCorpusManifest:
    """Versioned evidence envelope for a calibration corpus."""

    schema_version: str
    source_trace_ids: list[str]
    checkpoint_id: str
    teacher_id: str
    state_signature_version: str
    sample_count: int
    sampling_strategy: str
    inclusion_rules: dict[str, Any]
    exclusion_rules: dict[str, Any]
    coverage_fields: dict[str, Any]
    raw_production_frequency_weights: dict[str, float]
    bin_edges: list[float] | None
    calibration_split_hashes: list[str]
    test_split_hashes: list[str]
    no_test_leakage_asserted: bool
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_trace_ids": sorted(set(self.source_trace_ids)),
            "checkpoint_id": self.checkpoint_id,
            "teacher_id": self.teacher_id,
            "state_signature_version": self.state_signature_version,
            "sample_count": self.sample_count,
            "sampling_strategy": self.sampling_strategy,
            "inclusion_rules": self.inclusion_rules,
            "exclusion_rules": self.exclusion_rules,
            "coverage_fields": self.coverage_fields,
            "raw_production_frequency_weights": self.raw_production_frequency_weights,
            "bin_edges": self.bin_edges,
            "calibration_split_hashes": self.calibration_split_hashes,
            "test_split_hashes": self.test_split_hashes,
            "no_test_leakage_asserted": self.no_test_leakage_asserted,
            "created_at": self.created_at,
        }

    def assert_no_test_leakage(self) -> None:
        overlap = set(self.calibration_split_hashes) & set(self.test_split_hashes)
        if overlap:
            raise ValueError(
                f"calibration/test split overlap detected ({len(overlap)} hashes)"
            )
        self.no_test_leakage_asserted = True


def _sha256_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_to_sample(idx: int, record: Mapping[str, Any]) -> CalibrationSample:
    """Convert a grammar_decision trace row into a CalibrationSample."""
    trace_id = str(record.get("trajectory_id") or f"trace-{idx:08d}")
    sensitivity = record.get("sensitivity")
    sensitivity_score: float | None = None
    if isinstance(sensitivity, Mapping):
        values = [float(v) for v in sensitivity.values() if isinstance(v, (int, float))]
        if values:
            sensitivity_score = max(values)
    return CalibrationSample(
        trace_id=trace_id,
        state_fingerprint=str(record.get("state_fingerprint") or ""),
        state_signature_version=str(record.get("state_signature_version") or "1"),
        legal_action_ids=tuple(record.get("legal_action_ids") or []),
        selected_action_id=record.get("selected_action_id"),
        target_action_ids=tuple(record.get("target_action_ids") or []),
        top1_margin=_as_float(record.get("top1_margin")),
        posterior_entropy_bits=_as_float(record.get("posterior_entropy_bits")),
        scope_signature=str(record.get("scope_signature") or ""),
        template_signature=record.get("template_signature"),
        production_weight=1.0,
        bin_id=None,
        sensitivity_score=sensitivity_score,
        verification_outcome=record.get("verification_outcome"),
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_grammar_decision_traces(
    trace_dir: Path | str,
    *,
    checkpoint_id: str | None = None,
    state_signature_version: str | None = None,
) -> tuple[list[CalibrationSample], dict[str, Any], list[str]]:
    """Load and validate grammar_decision traces from a TraceStore.

    Returns ``(samples, coverage_report, violation_messages)``.
    """
    store = TraceStore(trace_dir)
    records: list[dict[str, Any]] = []
    violations: list[str] = []
    for idx, row in enumerate(store.iter_kind("grammar_decision")):
        rec_violations = grammar_trace_replay_violations([row])
        if rec_violations:
            violations.extend(f"{row.get('trajectory_id', idx)}: {v}" for v in rec_violations)
            continue
        if checkpoint_id and row.get("checkpoint_id") != checkpoint_id:
            continue
        if state_signature_version and row.get("state_signature_version") != state_signature_version:
            continue
        records.append(row)

    samples = [_record_to_sample(i, r) for i, r in enumerate(records)]
    coverage = grammar_trace_coverage_report(records)
    return samples, coverage, violations


def assign_production_weights(samples: Sequence[CalibrationSample]) -> list[CalibrationSample]:
    """Attach raw production frequency weights by selected action id."""
    selected = [s for s in samples if s.selected_action_id is not None]
    if not selected:
        return list(samples)
    counts = Counter(str(s.selected_action_id) for s in selected)
    total = len(selected)
    weights = {action: count / total for action, count in counts.items()}
    return [
        CalibrationSample(
            **{**s.to_dict(), "production_weight": weights.get(str(s.selected_action_id), 0.0)}
        )
        for s in samples
    ]


# ── sampling strategies ──────────────────────────────────────────────────────


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


def _weighted_choice_indices(
    weights: Sequence[float],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Sample with replacement using normalized weights."""
    total = sum(weights)
    if total <= 0:
        return rng.choices(range(len(weights)), k=n)
    return rng.choices(range(len(weights)), weights=weights, k=n)


def sample_random_production(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Random sampling weighted by raw production frequency."""
    if n > len(samples):
        raise ValueError(f"random_production requested {n} but only {len(samples)} records")
    weights = [s.production_weight for s in samples]
    return _weighted_choice_indices(weights, n, rng)


def sample_uniform_state(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Round-robin across states to maximize state coverage."""
    if not samples:
        return []
    by_state: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        by_state.setdefault(s.state_fingerprint, []).append(idx)
    order = list(by_state.values())
    for group in order:
        rng.shuffle(group)
    selected: list[int] = []
    pos = 0
    while len(selected) < n:
        added = False
        for group in order:
            if pos < len(group):
                selected.append(group[pos])
                added = True
                if len(selected) == n:
                    break
        pos += 1
        if not added:
            break
    if len(selected) < n:
        pool = list(range(len(samples)))
        extra = n - len(selected)
        selected.extend(rng.sample(pool, min(extra, len(pool))) if extra <= len(pool) else rng.choices(pool, k=extra))
    return selected


def sample_uniform_state_action(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Round-robin across (state, selected action) pairs."""
    if not samples:
        return []
    by_pair: dict[tuple[str, str], list[int]] = {}
    for idx, s in enumerate(samples):
        key = (s.state_fingerprint, str(s.selected_action_id or "__none__"))
        by_pair.setdefault(key, []).append(idx)
    order = list(by_pair.values())
    for group in order:
        rng.shuffle(group)
    selected: list[int] = []
    pos = 0
    while len(selected) < n:
        added = False
        for group in order:
            if pos < len(group):
                selected.append(group[pos])
                added = True
                if len(selected) == n:
                    break
        pos += 1
        if not added:
            break
    if len(selected) < n:
        pool = list(range(len(samples)))
        extra = n - len(selected)
        selected.extend(rng.sample(pool, min(extra, len(pool))) if extra <= len(pool) else rng.choices(pool, k=extra))
    return selected


def sample_scope_template_stratified(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Round-robin across (scope, template) bins."""
    if not samples:
        return []
    by_bin: dict[tuple[str, str], list[int]] = {}
    for idx, s in enumerate(samples):
        key = (s.scope_signature, s.template_signature or "__none__")
        by_bin.setdefault(key, []).append(idx)
    order = list(by_bin.values())
    for group in order:
        rng.shuffle(group)
    selected: list[int] = []
    pos = 0
    while len(selected) < n:
        added = False
        for group in order:
            if pos < len(group):
                selected.append(group[pos])
                added = True
                if len(selected) == n:
                    break
        pos += 1
        if not added:
            break
    if len(selected) < n:
        pool = list(range(len(samples)))
        extra = n - len(selected)
        selected.extend(rng.sample(pool, min(extra, len(pool))) if extra <= len(pool) else rng.choices(pool, k=extra))
    return selected


def sample_low_margin(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Prefer records with the smallest top-1 margin (most uncertain)."""
    scored = [(idx, s.top1_margin if s.top1_margin is not None else float("inf")) for idx, s in enumerate(samples)]
    scored.sort(key=lambda x: (x[1], rng.random()))
    selected = [idx for idx, _ in scored[:n]]
    if len(selected) < n:
        remaining = [idx for idx, _ in scored[n:]]
        extra = n - len(selected)
        selected.extend(rng.sample(remaining, min(extra, len(remaining))) if extra <= len(remaining) else rng.choices(remaining, k=extra))
    return selected


def sample_sensitivity_weighted(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Sample with replacement weighted by sensitivity score."""
    if not samples:
        return []
    weights = [s.sensitivity_score if s.sensitivity_score is not None else 0.0 for s in samples]
    return _weighted_choice_indices(weights, n, rng)


def sample_hybrid_coverage_margin(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
    *,
    bins: int = 3,
) -> list[int]:
    """Stratify by margin tercile, then uniformly sample states within each bin."""
    if not samples:
        return []
    margins = [s.top1_margin for s in samples if s.top1_margin is not None]
    if len(margins) < bins:
        return sample_uniform_state(samples, n, rng)
    sorted_margins = sorted(margins)
    edges = [sorted_margins[int(len(sorted_margins) * i / bins)] for i in range(1, bins)]

    def _bin_idx(margin: float | None) -> int:
        if margin is None:
            return bins - 1
        for i, edge in enumerate(edges):
            if margin <= edge:
                return i
        return bins - 1

    by_bin_state: dict[tuple[int, str], list[int]] = {}
    for idx, s in enumerate(samples):
        key = (_bin_idx(s.top1_margin), s.state_fingerprint)
        by_bin_state.setdefault(key, []).append(idx)
    order = list(by_bin_state.values())
    for group in order:
        rng.shuffle(group)
    selected: list[int] = []
    pos = 0
    while len(selected) < n:
        added = False
        for group in order:
            if pos < len(group):
                selected.append(group[pos])
                added = True
                if len(selected) == n:
                    break
        pos += 1
        if not added:
            break
    if len(selected) < n:
        pool = list(range(len(samples)))
        extra = n - len(selected)
        selected.extend(rng.sample(pool, min(extra, len(pool))) if extra <= len(pool) else rng.choices(pool, k=extra))
    return selected


def sample_active_counterexample(
    samples: Sequence[CalibrationSample],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Select only verified counterexamples; never fall back to test data."""
    candidates = [
        idx
        for idx, s in enumerate(samples)
        if s.verification_outcome == "counterexample"
    ]
    if not candidates:
        return []
    if n >= len(candidates):
        return candidates
    return rng.sample(candidates, n)


SAMPLING_REGISTRY: dict[str, Any] = {
    "random_production": sample_random_production,
    "uniform_state": sample_uniform_state,
    "uniform_state_action": sample_uniform_state_action,
    "scope_template_stratified": sample_scope_template_stratified,
    "low_margin": sample_low_margin,
    "sensitivity_weighted": sample_sensitivity_weighted,
    "hybrid_coverage_margin": sample_hybrid_coverage_margin,
    "active_counterexample": sample_active_counterexample,
}


def build_calibration_corpus(
    samples: Sequence[CalibrationSample],
    strategy: str,
    n: int,
    *,
    seed: int | None = None,
    test_split_hashes: Sequence[str] | None = None,
    checkpoint_id: str = "",
    teacher_id: str = "",
    inclusion_rules: dict[str, Any] | None = None,
    exclusion_rules: dict[str, Any] | None = None,
) -> tuple[CalibrationCorpusManifest, list[CalibrationSample]]:
    """Build a manifest and sampled calibration corpus."""
    if strategy not in SAMPLING_REGISTRY:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    if n <= 0:
        raise ValueError("sample count must be positive")
    if n > len(samples):
        raise ValueError(f"strategy {strategy!r} requested {n} samples but only {len(samples)} records")

    rng = _rng(seed)
    sampler = SAMPLING_REGISTRY[strategy]
    selected_indices = sampler(samples, n, rng)
    if strategy in PRIMARY_STRATEGIES and len(selected_indices) != n:
        raise RuntimeError(
            f"primary strategy {strategy!r} returned {len(selected_indices)} indices, expected {n}"
        )

    selected = [samples[i] for i in selected_indices]
    selected_hashes = [_sha256_json(s.to_dict()) for s in selected]

    # Coverage over the selected subset.
    selected_records = [s.to_dict() for s in selected]
    coverage = grammar_trace_coverage_report(selected_records)

    # Production weights over the whole input distribution.
    weighted = assign_production_weights(list(samples))
    freq: dict[str, float] = {}
    for s in weighted:
        if s.selected_action_id is not None:
            freq[str(s.selected_action_id)] = s.production_weight

    bin_edges: list[float] | None = None
    if strategy in {"low_margin", "hybrid_coverage_margin"}:
        margins = [s.top1_margin for s in samples if s.top1_margin is not None]
        if margins:
            lo, hi = min(margins), max(margins)
            if hi > lo:
                bin_edges = [lo + (hi - lo) * i / 5 for i in range(6)]

    manifest = CalibrationCorpusManifest(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        source_trace_ids=sorted({s.trace_id for s in samples}),
        checkpoint_id=checkpoint_id,
        teacher_id=teacher_id,
        state_signature_version=selected[0].state_signature_version if selected else "1",
        sample_count=len(selected),
        sampling_strategy=strategy,
        inclusion_rules=inclusion_rules or {},
        exclusion_rules=exclusion_rules or {},
        coverage_fields=coverage,
        raw_production_frequency_weights=freq,
        bin_edges=bin_edges,
        calibration_split_hashes=selected_hashes,
        test_split_hashes=list(test_split_hashes or []),
        no_test_leakage_asserted=False,
    )
    manifest.assert_no_test_leakage()
    return manifest, selected


# ── adaptation primitives ────────────────────────────────────────────────────


def calibrate_scales_ptq(
    weight: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """PTQ scale calibration: refit symmetric per-group scales for ``weight``."""
    return fake_quantize_weight(weight, fmt, group_size=group_size)


def _fake_quantize_with_ste(
    weight: torch.Tensor,
    fmt: QuantFormat,
    group_size: int | None = None,
) -> torch.Tensor:
    """Fake-quantize ``weight`` with a straight-through estimator for gradients.

    The existing ``fake_quantize`` marks its output non-differentiable; this
    wrapper re-attaches the quantized values to the leaf weight so SGD/QAT can
    update shadow weights.
    """
    with torch.no_grad():
        q, _, _ = fake_quantize_weight(weight, fmt, group_size=group_size)
    return weight + (q - weight).detach()


def _quantized_flat_head_logits(
    head: LocalFlatHead,
    hidden: torch.Tensor,
    legal_actions: list[str],
    fmt: QuantFormat,
) -> torch.Tensor:
    """Forward ``LocalFlatHead`` using fake-quantized action embeddings + STE."""
    import torch

    embeddings: list[torch.Tensor] = []
    for action in legal_actions:
        param = head.action_embeddings[action]
        q = _fake_quantize_with_ste(param, fmt, group_size=fmt.group_size)
        embeddings.append(q)
    stacked = torch.stack(embeddings, dim=0)
    return hidden @ stacked.T


def qat_reconstruct_local_scorer(
    head: LocalFlatHead,
    fmt: QuantFormat,
    batches: Sequence[tuple[torch.Tensor, list[str], torch.Tensor]],
    *,
    steps: int = 10,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Short QAT reconstruction on a local flat head using shadow weights + STE.

    ``batches`` is a sequence of ``(hidden, legal_actions, teacher_logits)``.
    The teacher distribution is derived via softmax over ``teacher_logits``
    restricted to the legal actions. Only action-embedding parameters are
    updated.
    """
    import torch
    import torch.nn.functional as F

    parameters = list(head.action_embeddings.values())
    if not parameters:
        return {"status": "no_parameters", "steps": 0, "final_loss": None}
    optimizer = torch.optim.SGD(parameters, lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        step_loss = 0.0
        count = 0
        for hidden, legal_actions, teacher_logits in batches:
            optimizer.zero_grad()
            student_logits = _quantized_flat_head_logits(head, hidden, legal_actions, fmt)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction="batchmean",
            )
            loss.backward()
            optimizer.step()
            step_loss += float(loss.item())
            count += 1
        losses.append(step_loss / max(count, 1))
    return {
        "status": "ok",
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
    }


def kl_objective(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """KL divergence from student to teacher distribution."""
    import torch.nn.functional as F

    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs,
        reduction="batchmean",
    )


def ranking_objective(
    student_logits: torch.Tensor,
    target_index: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge-style ranking loss pushing the target above all competitors."""
    import torch

    target_score = student_logits.gather(-1, target_index.unsqueeze(-1)).squeeze(-1)
    mask = torch.ones_like(student_logits, dtype=torch.bool)
    mask.scatter_(-1, target_index.unsqueeze(-1), False)
    best_competitor = student_logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
    return (margin - (target_score - best_competitor)).clamp_min(0).mean()


def margin_objective(
    student_margin: torch.Tensor,
    teacher_margin: torch.Tensor,
) -> torch.Tensor:
    """MSE between student and teacher decision margins."""
    import torch.nn.functional as F

    return F.mse_loss(student_margin, teacher_margin)


def path_ranking_objective(
    student_logits: torch.Tensor,
    legal_actions: list[str],
    target_semantics: Sequence[str],
) -> torch.Tensor:
    """Ranking loss over actions grouped by target semantics string."""
    import torch

    if not legal_actions or len(target_semantics) != len(legal_actions):
        return torch.tensor(0.0)
    groups: dict[str, list[int]] = {}
    for idx, sem in enumerate(target_semantics):
        groups.setdefault(sem, []).append(idx)
    loss = torch.tensor(0.0)
    count = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group_logits = student_logits[:, indices]
        target = torch.arange(len(indices), device=group_logits.device)
        loss = loss + torch.nn.functional.cross_entropy(group_logits, target)
        count += 1
    return loss / max(count, 1)


def mixed_task_distillation_objective(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    target_index: torch.Tensor,
    *,
    task_weight: float = 0.5,
    distill_weight: float = 0.5,
) -> torch.Tensor:
    """Combine hard-target CE with teacher KL."""
    import torch.nn.functional as F

    ce = F.cross_entropy(student_logits, target_index)
    kl = kl_objective(student_logits, teacher_probs)
    return task_weight * ce + distill_weight * kl


# ── orchestration ────────────────────────────────────────────────────────────


def run_calibration(
    trace_dir: Path | str,
    strategy: str,
    n: int,
    *,
    checkpoint_id: str = "",
    teacher_id: str = "",
    seed: int | None = None,
    test_split_hashes: Sequence[str] | None = None,
    inclusion_rules: dict[str, Any] | None = None,
    exclusion_rules: dict[str, Any] | None = None,
) -> tuple[CalibrationCorpusManifest, list[CalibrationSample], dict[str, Any]]:
    """High-level entry point: load traces, sample, build manifest, return evidence."""
    samples, coverage, violations = load_grammar_decision_traces(
        trace_dir,
        checkpoint_id=checkpoint_id or None,
    )
    if not samples:
        raise ValueError(f"no grammar_decision traces found in {trace_dir}")

    manifest, selected = build_calibration_corpus(
        samples,
        strategy,
        n,
        seed=seed,
        test_split_hashes=test_split_hashes,
        checkpoint_id=checkpoint_id,
        teacher_id=teacher_id,
        inclusion_rules=inclusion_rules,
        exclusion_rules=exclusion_rules,
    )
    evidence = {
        "trace_store_violations": violations,
        "full_coverage": coverage,
        "selected_count": len(selected),
        "unique_selected_states": len({s.state_fingerprint for s in selected}),
    }
    return manifest, selected, evidence
