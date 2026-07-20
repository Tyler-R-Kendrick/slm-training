"""SLM-168 (SDE2-01): public structured contract-index pointer fixture.

Wiring-only default-off harness that exercises explicit contract-index pointer
candidate sets and a dynamic pointer scorer across request modes.  No live
decode path is changed and no model is trained.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.models.dynamic_pointer_scorer import (
    DynamicPointerScorer,
    DynamicPointerScorerConfig,
    PointerCandidate,
    PointerCandidateSet,
    PointerDecision,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "POINTER_MODES",
    "CANDIDATE_SOURCES",
    "ContractPointerArm",
    "ContractPointerMetrics",
    "ContractPointerReport",
    "build_cells",
    "validate_manifest",
    "build_candidate_set",
    "score_pointer_decision",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde2-01-v1"
MATRIX_SET = "slm168_public_structured_contract_pointer"
EXPERIMENT_ID = "slm168-public-structured-contract-pointer"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_D_MODEL = 64

POINTER_MODES = ("legacy_tokens", "dynamic_head")
CANDIDATE_SOURCES = ("structured_contract", "authored_only", "inventory_in_prompt")
ARM_NAMES = (
    "legacy_inventory_in_prompt",
    "legacy_no_inventory",
    "dynamic_structured_contract",
    "dynamic_authored_only",
    "dynamic_inventory_in_prompt",
    "dynamic_permuted_order",
    "dynamic_hidden_text",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContractPointerArm:
    """One contract-pointer experiment arm."""

    arm_id: str
    arm_name: str
    pointer_mode: str
    candidate_source: str
    seed: int
    d_model: int
    pointer_hidden_dim: int
    pointer_heads: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractPointerArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            pointer_mode=str(data["pointer_mode"]),
            candidate_source=str(data["candidate_source"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            pointer_hidden_dim=int(data["pointer_hidden_dim"]),
            pointer_heads=int(data["pointer_heads"]),
        )


@dataclass(frozen=True)
class ContractPointerMetrics:
    """Per-arm, per-seed synthetic fixture metrics."""

    arm_id: str
    arm_name: str
    pointer_mode: str
    candidate_source: str
    seed: int
    d_model: int
    pointer_hidden_dim: int
    pointer_heads: int
    candidate_discovery_recall: float
    pointer_top1_accuracy: float
    pointer_mrr: float
    binding_fidelity: float
    meaningful_program_rate: float
    rare_component_recall: float
    parse_validity_rate: float
    permutation_equivariant: bool
    fail_closed_rate: float
    wall_seconds: float
    notes: list[str] = field(default_factory=list)
    decisions: list[PointerDecision] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["decisions"] = [d.to_dict() for d in self.decisions]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractPointerMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            pointer_mode=str(data["pointer_mode"]),
            candidate_source=str(data["candidate_source"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            pointer_hidden_dim=int(data["pointer_hidden_dim"]),
            pointer_heads=int(data["pointer_heads"]),
            candidate_discovery_recall=float(data["candidate_discovery_recall"]),
            pointer_top1_accuracy=float(data["pointer_top1_accuracy"]),
            pointer_mrr=float(data["pointer_mrr"]),
            binding_fidelity=float(data["binding_fidelity"]),
            meaningful_program_rate=float(data["meaningful_program_rate"]),
            rare_component_recall=float(data["rare_component_recall"]),
            parse_validity_rate=float(data["parse_validity_rate"]),
            permutation_equivariant=bool(data["permutation_equivariant"]),
            fail_closed_rate=float(data["fail_closed_rate"]),
            wall_seconds=float(data["wall_seconds"]),
            notes=list(data.get("notes", [])),
            decisions=[
                PointerDecision.from_dict(d) for d in data.get("decisions", [])
            ],
        )


@dataclass(frozen=True)
class ContractPointerReport:
    """Full fixture report for SLM-168."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[ContractPointerArm, ...]
    rows: list[ContractPointerMetrics]
    arm_means: dict[str, dict[str, float]]
    disposition: str
    disposition_rationale: str
    dependency_caveats: list[str]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cells": [cell.to_dict() for cell in self.cells],
            "rows": [row.to_dict() for row in self.rows],
            "arm_means": {k: dict(v) for k, v in self.arm_means.items()},
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "dependency_caveats": list(self.dependency_caveats),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContractPointerReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm168_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "Explicit contract-index pointer supervision preserves binding fidelity "
                "when prompt inventory is removed, because pointer identity is learned "
                "relative to a live request-visible candidate set.",
            ),
            falsifier=data.get(
                "falsifier",
                "The dynamic pointer arm cannot beat the legacy slot-token representation "
                "without prompt inventory, or gains vanish under candidate-order permutation.",
            ),
            cells=tuple(ContractPointerArm.from_dict(c) for c in data.get("cells", [])),
            rows=[ContractPointerMetrics.from_dict(r) for r in data.get("rows", [])],
            arm_means={
                k: dict(v) for k, v in data.get("arm_means", {}).items()
            },
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _hash_float(payload: str, span: float = 1.0) -> float:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = int(digest[:16], 16) / (2 ** 64)
    return (normalized * 2.0 - 1.0) * span


def _arm_label(arm_name: str, seed: int) -> str:
    return f"{arm_name}__s{seed}"


def _arm_config(arm_name: str) -> dict[str, Any]:
    if arm_name == "legacy_inventory_in_prompt":
        return {"pointer_mode": "legacy_tokens", "candidate_source": "inventory_in_prompt"}
    if arm_name == "legacy_no_inventory":
        return {"pointer_mode": "legacy_tokens", "candidate_source": "structured_contract"}
    if arm_name == "dynamic_structured_contract":
        return {"pointer_mode": "dynamic_head", "candidate_source": "structured_contract"}
    if arm_name == "dynamic_authored_only":
        return {"pointer_mode": "dynamic_head", "candidate_source": "authored_only"}
    if arm_name == "dynamic_inventory_in_prompt":
        return {"pointer_mode": "dynamic_head", "candidate_source": "inventory_in_prompt"}
    if arm_name == "dynamic_permuted_order":
        return {"pointer_mode": "dynamic_head", "candidate_source": "structured_contract"}
    if arm_name == "dynamic_hidden_text":
        return {"pointer_mode": "dynamic_head", "candidate_source": "structured_contract"}
    raise ValueError(f"unknown arm_name: {arm_name!r}")


def build_candidate_set(
    prompt: str,
    *,
    source: str,
    seed: int,
    include_gold: bool = True,
) -> PointerCandidateSet:
    """Build a deterministic request-visible candidate set for a prompt.

    The candidate set is constructed from inference-available fields only.  It
    deliberately does not inspect any gold AST or evaluator-only placeholders.
    """
    rng = random.Random(seed)
    base_slots = [":hero.title", ":hero.body", ":hero.actions", ":hero.footer"]
    runtime = [":user.name", ":user.email", ":session.id"]
    schema = ["Hero", "Button", "Input", "Stack"]

    candidates: list[PointerCandidate] = []
    if source == "inventory_in_prompt":
        # Full inventory available in prose and public contract.
        for s in base_slots + runtime:
            candidates.append(
                PointerCandidate(
                    stable_id=s,
                    display_text=s,
                    kind="slot",
                    type_name="string",
                    provenance="authored_prompt",
                )
            )
    elif source == "structured_contract":
        # Public structured contract only; prose inventory omitted.
        for s in base_slots:
            candidates.append(
                PointerCandidate(
                    stable_id=s,
                    display_text=s,
                    kind="slot",
                    type_name="string",
                    provenance="request_contract",
                )
            )
    elif source == "authored_only":
        # Candidates extracted from prompt prose only.
        prompt_text = prompt.lower()
        for s in base_slots:
            # Use the last component of the slot (e.g., "title" from ":hero.title").
            last_component = s.rsplit(".", 1)[-1].lstrip(":")
            if last_component in prompt_text:
                candidates.append(
                    PointerCandidate(
                        stable_id=s,
                        display_text=s,
                        kind="slot",
                        type_name="string",
                        provenance="authored_prompt",
                    )
                )
    else:
        raise ValueError(f"unknown candidate source: {source!r}")

    # Add a few schema entities visible to the compiler scope.
    for name in rng.sample(schema, k=min(2, len(schema))):
        candidates.append(
            PointerCandidate(
                stable_id=f"schema:{name}",
                display_text=name,
                kind="schema_entity",
                type_name="component",
                provenance="compiler_scope",
            )
        )

    # Optionally drop the gold candidate to test fail-closed behavior.
    if not include_gold and candidates:
        candidates = candidates[1:]

    return PointerCandidateSet(
        candidates=tuple(candidates),
        permitted_sources=(source, "compiler_scope"),
        manifest_hash=hashlib.sha256(
            json.dumps([c.to_dict() for c in candidates], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
    )


def score_pointer_decision(
    arm: ContractPointerArm,
    candidate_set: PointerCandidateSet,
    gold_stable_id: str,
    state_signature: str,
    *,
    permute: bool = False,
    hide_text: bool = False,
) -> PointerDecision:
    """Score one pointer decision for a candidate set."""
    start = time.perf_counter()
    config = DynamicPointerScorerConfig(
        pointer_mode=arm.pointer_mode,  # type: ignore[arg-type]
        pointer_candidate_source=arm.candidate_source,  # type: ignore[arg-type]
        d_model=arm.d_model,
        pointer_hidden_dim=arm.pointer_hidden_dim,
        pointer_heads=arm.pointer_heads,
    )
    scorer = DynamicPointerScorer(config)

    candidates = list(candidate_set.candidates)
    if permute:
        rng = random.Random(arm.seed)
        rng.shuffle(candidates)
    if hide_text:
        candidates = [
            PointerCandidate(
                stable_id=c.stable_id,
                display_text="",
                kind=c.kind,
                type_name=c.type_name,
                provenance=c.provenance,
            )
            for c in candidates
        ]

    final_set = PointerCandidateSet(
        candidates=tuple(candidates),
        permitted_sources=candidate_set.permitted_sources,
        manifest_hash=candidate_set.manifest_hash,
    )

    state_vec = torch.randn(arm.d_model, dtype=torch.float32)
    state_vec = state_vec / state_vec.norm()

    gold_index = final_set.index(gold_stable_id)
    mask = torch.ones(len(final_set), dtype=torch.bool)
    with torch.no_grad():
        log_probs = scorer.score(state_vec, final_set, mask=mask)
    scores = log_probs.exp().tolist()
    selected_index = int(torch.argmax(log_probs).item())

    elapsed = time.perf_counter() - start
    return PointerDecision(
        state_signature=state_signature,
        candidate_set_hash=final_set.manifest_hash,
        gold_index=gold_index,
        selected_index=selected_index,
        scores=tuple(scores),
        mask=tuple(mask.tolist()),
        pointer_mode=arm.pointer_mode,
        candidate_source=arm.candidate_source,
        latency_seconds=elapsed,
    )


def _simulate_cell(arm: ContractPointerArm) -> ContractPointerMetrics:
    """Run the fixture for one arm/seed."""
    start = time.perf_counter()
    rng = random.Random(arm.seed)
    prompts = (
        "Build a hero section with title, body, and actions.",
        "Create a user profile card with name and email.",
        "Design a settings panel with session id and footer.",
    )
    decisions: list[PointerDecision] = []
    gold_ids = [":hero.title", ":hero.body", ":user.name", ":session.id"]

    for i, prompt in enumerate(prompts):
        # Include gold for the first two prompts; fail-closed for the third.
        include_gold = i < 2
        candidate_set = build_candidate_set(
            prompt,
            source=arm.candidate_source,
            seed=arm.seed + i,
            include_gold=include_gold,
        )
        gold_id = rng.choice(gold_ids)
        decision = score_pointer_decision(
            arm,
            candidate_set,
            gold_id,
            state_signature=f"state_{arm.arm_id}_{i}",
            permute=(arm.arm_name == "dynamic_permuted_order"),
            hide_text=(arm.arm_name == "dynamic_hidden_text"),
        )
        decisions.append(decision)

    n_gold = sum(1 for d in decisions if d.gold_index is not None)
    top1_hits = sum(
        1 for d in decisions if d.gold_index is not None and d.selected_index == d.gold_index
    )
    mrr_sum = sum(
        1.0 / (d.selected_index + 1) if d.gold_index is not None else 0.0
        for d in decisions
    )
    discovery_recall = n_gold / max(1, len(decisions))

    # Synthetic quality is driven by the arm configuration.
    base_quality = {
        "legacy_inventory_in_prompt": 0.85,
        "legacy_no_inventory": 0.45,
        "dynamic_structured_contract": 0.78,
        "dynamic_authored_only": 0.55,
        "dynamic_inventory_in_prompt": 0.82,
        "dynamic_permuted_order": 0.76,
        "dynamic_hidden_text": 0.68,
    }[arm.arm_name]

    binding_fidelity = _clamp(
        base_quality + _hash_float(f"bf:{arm.arm_id}", span=0.02)
    )
    meaningful_program_rate = _clamp(
        binding_fidelity - 0.05 + _hash_float(f"mp:{arm.arm_id}", span=0.02)
    )
    rare_component_recall = _clamp(
        meaningful_program_rate - 0.03 + _hash_float(f"rcr:{arm.arm_id}", span=0.02)
    )
    parse_validity_rate = _clamp(
        meaningful_program_rate + 0.08 + _hash_float(f"pv:{arm.arm_id}", span=0.02)
    )

    # Permutation equivariance: dynamic head should be robust to candidate order.
    permutation_equivariant = arm.arm_name != "legacy_tokens"

    fail_closed_hits = sum(
        1 for d in decisions if d.gold_index is None and d.selected_index == 0
    )
    fail_closed_rate = fail_closed_hits / max(1, len(decisions) - n_gold)

    elapsed = time.perf_counter() - start
    wall_seconds = _clamp(
        elapsed + 0.005 * len(decisions) + _hash_float(f"wall:{arm.arm_id}", span=0.02),
        low=0.001,
        high=10.0,
    )

    notes = [
        f"pointer_mode={arm.pointer_mode}",
        f"candidate_source={arm.candidate_source}",
        "fixture-only: synthetic pointer supervision comparison",
    ]
    if arm.arm_name == "dynamic_permuted_order":
        notes.append("candidate order permuted per example")
    if arm.arm_name == "dynamic_hidden_text":
        notes.append("display text hidden, kind/type retained")

    return ContractPointerMetrics(
        arm_id=arm.arm_id,
        arm_name=arm.arm_name,
        pointer_mode=arm.pointer_mode,
        candidate_source=arm.candidate_source,
        seed=arm.seed,
        d_model=arm.d_model,
        pointer_hidden_dim=arm.pointer_hidden_dim,
        pointer_heads=arm.pointer_heads,
        candidate_discovery_recall=discovery_recall,
        pointer_top1_accuracy=top1_hits / max(1, n_gold),
        pointer_mrr=mrr_sum / max(1, n_gold),
        binding_fidelity=binding_fidelity,
        meaningful_program_rate=meaningful_program_rate,
        rare_component_recall=rare_component_recall,
        parse_validity_rate=parse_validity_rate,
        permutation_equivariant=permutation_equivariant,
        fail_closed_rate=fail_closed_rate,
        wall_seconds=wall_seconds,
        notes=notes,
        decisions=decisions,
    )


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    *,
    d_model: int = _DEFAULT_D_MODEL,
    pointer_hidden_dim: int = 256,
    pointer_heads: int = 4,
) -> tuple[ContractPointerArm, ...]:
    """Build the 7 arms × seeds contract-pointer cells."""
    cells: list[ContractPointerArm] = []
    for seed in seeds:
        for arm_name in ARM_NAMES:
            cfg = _arm_config(arm_name)
            cells.append(
                ContractPointerArm(
                    arm_id=_arm_label(arm_name, seed),
                    arm_name=arm_name,
                    pointer_mode=cfg["pointer_mode"],
                    candidate_source=cfg["candidate_source"],
                    seed=seed,
                    d_model=d_model,
                    pointer_hidden_dim=pointer_hidden_dim,
                    pointer_heads=pointer_heads,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[ContractPointerArm, ...]) -> list[str]:
    """Validate the contract-pointer manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    for cell in cells:
        if cell.arm_id in seen:
            errors.append(f"duplicate arm_id: {cell.arm_id}")
        seen.add(cell.arm_id)
        if cell.pointer_mode not in POINTER_MODES:
            errors.append(f"{cell.arm_id}: invalid pointer_mode {cell.pointer_mode!r}")
        if cell.candidate_source not in CANDIDATE_SOURCES:
            errors.append(
                f"{cell.arm_id}: invalid candidate_source {cell.candidate_source!r}"
            )
    return errors


def _arm_means(rows: list[ContractPointerMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "candidate_discovery_recall",
            "pointer_top1_accuracy",
            "pointer_mrr",
            "binding_fidelity",
            "meaningful_program_rate",
            "rare_component_recall",
            "parse_validity_rate",
            "fail_closed_rate",
            "wall_seconds",
        ):
            bucket.setdefault(key, []).append(float(getattr(row, key)))
    return {
        arm: {key: statistics.mean(values) for key, values in metrics.items()}
        for arm, metrics in grouped.items()
    }


def resolve_disposition(
    arm_means: dict[str, dict[str, float]]
) -> tuple[str, str]:
    """Return (disposition, rationale) from the per-arm means."""
    legacy_inventory = arm_means.get("legacy_inventory_in_prompt", {}).get("binding_fidelity", 0.0)
    legacy_no_inv = arm_means.get("legacy_no_inventory", {}).get("binding_fidelity", 0.0)
    dynamic_structured = arm_means.get("dynamic_structured_contract", {}).get("binding_fidelity", 0.0)
    dynamic_inventory = arm_means.get("dynamic_inventory_in_prompt", {}).get("binding_fidelity", 0.0)
    dynamic_permuted = arm_means.get("dynamic_permuted_order", {}).get("binding_fidelity", 0.0)

    if dynamic_structured < legacy_no_inv + 0.05:
        return (
            "pointer_not_better_than_legacy_no_inventory",
            "The dynamic pointer head does not improve over legacy slot tokens "
            "without prompt inventory; explicit pointer supervision is not sufficient "
            "in this fixture.",
        )
    if dynamic_permuted < dynamic_structured - 0.10:
        return (
            "order_sensitive_pointer",
            "Permuting candidate order materially degrades pointer fidelity, so the "
            "scorer is not permutation-equivariant.",
        )
    if dynamic_structured >= 0.90 and dynamic_structured - legacy_no_inv >= 0.10:
        return (
            "contract_conditioned_pointer_works",
            "The dynamic pointer head preserves high binding fidelity with a structured "
            "public contract and no prompt inventory, and remains permutation-robust.",
        )
    if dynamic_inventory >= legacy_inventory - 0.05:
        return (
            "inventory_equivalent",
            "The dynamic pointer head matches legacy inventory-in-prompt fidelity when "
            "inventory is available, but authored-only discovery remains weaker.",
        )
    return (
        "inconclusive",
        "The pointer matrix pattern is mixed; additional real-model measurements are "
        "needed to falsify H8.",
    )


def run_fixture_campaign(
    cells: tuple[ContractPointerArm, ...] | None = None,
    *,
    run_id: str = "slm168-public-structured-contract-pointer",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
) -> ContractPointerReport:
    """Run the SLM-168 contract-pointer fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    rows = [_simulate_cell(cell) for cell in cells]
    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    hypothesis = (
        "Explicit contract-index pointer supervision can preserve binding fidelity "
        "with prompt inventory removed because pointer identity is learned relative "
        "to a live request-visible candidate set rather than as a global vocabulary item."
    )
    falsifier = (
        "The dynamic pointer arm cannot beat the current slot-token representation "
        "without prompt inventory, or gains vanish under candidate-order permutation."
    )

    report = ContractPointerReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=hypothesis,
        falsifier=falsifier,
        cells=cells,
        rows=rows,
        arm_means=means,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on SLM-161 decode-scaffolding conventions for inventory factors.",
            "Depends on SLM-162 metric-gaming suite for inventory-free stress cases.",
            "Dynamic scorer is not wired into live TwoTower decode.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm168_public_structured_contract_pointer",
            "model.twotower",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm168_public_structured_contract_pointer_report.json")
    return report


def render_markdown(report: ContractPointerReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-168 (SDE2-01): public structured contract-index pointer fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no live decode path "
        "was changed, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Pointer arms",
        "",
        "| arm_id | arm_name | pointer_mode | candidate_source | seed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.arm_id} | {cell.arm_name} | {cell.pointer_mode} | "
            f"{cell.candidate_source} | {cell.seed} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arm_id | arm_name | seed | discovery_recall | pointer_top1 | pointer_mrr | "
            "binding_fidelity | meaningful_rate | rare_recall | parse_validity | fail_closed | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.seed} | "
            f"{row.candidate_discovery_recall:.3f} | {row.pointer_top1_accuracy:.3f} | "
            f"{row.pointer_mrr:.3f} | {row.binding_fidelity:.3f} | "
            f"{row.meaningful_program_rate:.3f} | {row.rare_component_recall:.3f} | "
            f"{row.parse_validity_rate:.3f} | {row.fail_closed_rate:.3f} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | discovery_recall | pointer_top1 | pointer_mrr | binding_fidelity | "
            "meaningful_rate | rare_recall | parse_validity | fail_closed |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        lines.append(
            f"| {arm_name} | {m.get('candidate_discovery_recall', 0.0):.3f} | "
            f"{m.get('pointer_top1_accuracy', 0.0):.3f} | {m.get('pointer_mrr', 0.0):.3f} | "
            f"{m.get('binding_fidelity', 0.0):.3f} | {m.get('meaningful_program_rate', 0.0):.3f} | "
            f"{m.get('rare_component_recall', 0.0):.3f} | {m.get('parse_validity_rate', 0.0):.3f} | "
            f"{m.get('fail_closed_rate', 0.0):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The pointer candidate "
            "contract, dynamic scorer variants, and synthetic metrics are exercised "
            "over deterministic inputs, but no real model was trained or evaluated. "
            "The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` "
            "until a trained scorer and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Metrics are generated by a deterministic simulator, not a trained model.",
            "- The dynamic scorer is not wired into live TwoTower decode.",
            "- Candidate discovery is simplified; real candidate extraction belongs to the "
            "  request/compiler pipeline.",
            "- No content floor, hidden slot contract, best-of-N, or retry is used.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm168_public_structured_contract_pointer_fixture --mode plan-only",
            "python -m scripts.run_slm168_public_structured_contract_pointer_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
