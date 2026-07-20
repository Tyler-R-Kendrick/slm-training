"""SLM-135 EFS4-01 trailed-assumptions ablation fixture harness.

A wiring-only benchmark adapter that implements four deterministic finite-domain
search policies on a hand-written closed fixture:

* ``certified_trail``   — production-style reversible decisions + dependency tracking
* ``monotone_proposal`` — intentionally unsafe: proposal-derived deductions persist
                          after the decision that created them is rolled back
* ``partial_retract``   — retract the decision value but keep its deductions
* ``certified_only_no_branch`` — exact closure only, no reversible branching

No production solver is changed.  The fixture is intentionally tiny so the
formal boundary can be regression-tested without a frontier campaign.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp

__all__ = [
    "AblationPolicy",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "Slm135Arm",
    "Slm135Manifest",
    "Slm135Report",
    "Slm135Row",
    "build_ablation_fixture",
    "build_manifest",
    "render_markdown",
    "run_ablation_search",
    "run_fixture_matrix",
]

MATRIX_VERSION = "efs4-01-v1"
MATRIX_SET = "slm135-trailed-assumptions"


class AblationPolicy(str, Enum):
    CERTIFIED_TRAIL = "certified_trail"
    MONOTONE_PROPOSAL = "monotone_proposal"
    PARTIAL_RETRACT = "partial_retract"
    CERTIFIED_ONLY_NO_BRANCH = "certified_only_no_branch"


@dataclass(frozen=True)
class FixtureValue:
    """One discrete candidate value."""

    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class FixtureHole:
    """A finite hole and its live candidate values."""

    hole_id: str
    values: tuple[FixtureValue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "values": [v.to_dict() for v in self.values],
        }


@dataclass(frozen=True)
class FixtureState:
    """Immutable finite-domain state used by the ablation fixture.

    The fingerprint intentionally excludes mutable decision metadata so that
    ``pre_state.fingerprint`` is reproducible across backtracks.
    """

    holes: tuple[FixtureHole, ...]
    decision_level: int = 0
    parent_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_fingerprint",
            self._compute_fingerprint(),
        )

    @property
    def fingerprint(self) -> str:
        return object.__getattribute__(self, "_fingerprint")

    def _compute_fingerprint(self) -> str:
        payload = {
            "holes": [
                {
                    "hole_id": h.hole_id,
                    "values": sorted(v.name for v in h.values),
                }
                for h in self.holes
            ]
        }
        return hash_json(payload)

    def domain(self, hole_id: str) -> FixtureHole | None:
        for h in self.holes:
            if h.hole_id == hole_id:
                return h
        return None

    def with_decision(self, hole_id: str, value: FixtureValue) -> "FixtureState":
        """Commit a single value for a hole and advance the decision level."""
        new_holes = tuple(
            FixtureHole(h.hole_id, (value,)) if h.hole_id == hole_id else h
            for h in self.holes
        )
        return FixtureState(
            holes=new_holes,
            decision_level=self.decision_level + 1,
            parent_fingerprint=self.fingerprint,
        )

    def remove_value(self, hole_id: str, value: FixtureValue) -> "FixtureState":
        """Remove one candidate from a hole (used for deductions and nogoods)."""
        new_holes: list[FixtureHole] = []
        for h in self.holes:
            if h.hole_id == hole_id:
                remaining = tuple(v for v in h.values if v.name != value.name)
                new_holes.append(FixtureHole(h.hole_id, remaining))
            else:
                new_holes.append(h)
        return FixtureState(
            holes=tuple(new_holes),
            decision_level=self.decision_level,
            parent_fingerprint=self.parent_fingerprint,
        )

    @property
    def is_bottom(self) -> bool:
        return any(len(h.values) == 0 for h in self.holes)

    @property
    def is_fully_assigned(self) -> bool:
        return all(len(h.values) == 1 for h in self.holes) and not self.is_bottom

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "decision_level": self.decision_level,
            "parent_fingerprint": self.parent_fingerprint,
            "holes": [h.to_dict() for h in self.holes],
        }


def hash_json(payload: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class CertifiedDeduction:
    """A certificate-backed removal produced by exact closure."""

    hole_id: str
    removed: FixtureValue
    depends_on: tuple[str, ...]
    certificate: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "removed": self.removed.to_dict(),
            "depends_on": list(self.depends_on),
            "certificate": self.certificate,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """One reversible search decision."""

    decision_id: str
    level: int
    hole_id: str
    chosen: FixtureValue
    alternatives: tuple[FixtureValue, ...]
    before_fingerprint: str
    after_fingerprint: str
    policy: AblationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "level": self.level,
            "hole_id": self.hole_id,
            "chosen": self.chosen.to_dict(),
            "alternatives": [v.to_dict() for v in self.alternatives],
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "policy": self.policy.value,
        }


@dataclass(frozen=True)
class NogoodRecord:
    """Request-local conflict record (not a certified deduction)."""

    hole_id: str
    value: FixtureValue
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "value": self.value.to_dict(),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AblationResult:
    """Outcome of one policy run."""

    policy: AblationPolicy
    status: str
    terminal: tuple[tuple[str, str], ...]
    decisions: tuple[DecisionRecord, ...]
    deductions: tuple[CertifiedDeduction, ...]
    nogoods: tuple[NogoodRecord, ...]
    backtracks: int
    nodes: int
    false_prune: bool
    unknown_violation: bool
    leaked_deductions: tuple[CertifiedDeduction, ...]
    restored_fingerprint: str | None
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "status": self.status,
            "terminal": [list(t) for t in self.terminal],
            "decisions": [d.to_dict() for d in self.decisions],
            "deductions": [d.to_dict() for d in self.deductions],
            "nogoods": [n.to_dict() for n in self.nogoods],
            "backtracks": self.backtracks,
            "nodes": self.nodes,
            "false_prune": self.false_prune,
            "unknown_violation": self.unknown_violation,
            "leaked_deductions": [d.to_dict() for d in self.leaked_deductions],
            "restored_fingerprint": self.restored_fingerprint,
            "stop_reason": self.stop_reason,
        }


@dataclass
class _Frame:
    pre_state: FixtureState
    post_state: FixtureState
    hole_id: str
    chosen: FixtureValue
    remaining: list[FixtureValue]
    level: int
    deductions: list[CertifiedDeduction]


@dataclass(frozen=True)
class Slm135Arm:
    """One ablation-policy arm."""

    arm_id: str
    policy: AblationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "policy": self.policy.value}


@dataclass(frozen=True)
class Slm135Manifest:
    """Preregistered manifest for the SLM-135 ablation fixture."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    hypothesis: str = (
        "When a locally legal proposal is wrong, a monotone proposal-contingent "
        "state permanently removes valid alternatives that depended on the failed "
        "assumption; a trailed controller retracts those facts and recovers with "
        "zero false certified prunes."
    )
    falsifier: str = (
        "Either the production architecture never places proposal-derived facts in "
        "irreversible state, or on activated cases both policies recover identical "
        "support sets while trailing adds only measurable cost."
    )
    arms: tuple[Slm135Arm, ...] = ()
    seeds: tuple[int, ...] = (0, 1, 2)
    claim_class: str = "wiring"
    status: str = "not_run"
    activation_gate: str = (
        "Frontier/natural cases run only when an earlier readout establishes an "
        "activated branching/recovery regime. This fixture uses only the closed "
        "finite benchmark and injected microcases."
    )

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = [a.to_dict() for a in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Slm135Row:
    """Measured result for one (policy, seed) fixture run."""

    arm_id: str
    policy: AblationPolicy
    seed: int
    status: str
    result: AblationResult
    decision_count: int
    backtrack_count: int
    false_prune: bool
    leaked_deduction_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "policy": self.policy.value,
            "seed": self.seed,
            "status": self.status,
            "result": self.result.to_dict(),
            "decision_count": self.decision_count,
            "backtrack_count": self.backtrack_count,
            "false_prune": self.false_prune,
            "leaked_deduction_count": self.leaked_deduction_count,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Slm135Report:
    """Full ablation fixture report."""

    matrix_set: str
    matrix_version: str
    run_id: str
    status: str
    manifest: Slm135Manifest
    rows: list[Slm135Row]
    verdict: str
    claim_class: str = "wiring"
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "verdict": self.verdict,
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _accepted_terminals() -> set[tuple[str, str]]:
    """Ground-truth accepted terminals for the closed fixture."""
    return {("a2", "c2")}


def _check_terminal(state: FixtureState) -> tuple[bool, tuple[tuple[str, str], ...]]:
    if not state.is_fully_assigned:
        return False, ()
    assignment = tuple(
        (h.values[0].name for h in state.holes if h.hole_id == name).__next__()
        for name in ("A", "C")
    )
    terminal = (assignment[0], assignment[1])
    accepted = terminal in _accepted_terminals()
    return accepted, (terminal,) if accepted else ()


def _apply_deductions(
    state: FixtureState,
) -> tuple[FixtureState, list[CertifiedDeduction]]:
    """Deterministic exact-closure step for the fixture.

    The only certified deductions are:
    * if A=a1 then C=c2 is unsupported (certificate ``cert-a1-not-c2``)
    * if A=a2 then C=c1 is unsupported (certificate ``cert-a2-not-c1``)

    These are assumption-dependent: they are removed on rollback in the
    ``certified_trail`` policy and retained in the unsafe policies.
    """
    deductions: list[CertifiedDeduction] = []
    a_domain = state.domain("A")
    c_domain = state.domain("C")
    if a_domain is None or c_domain is None:
        return state, deductions
    if len(a_domain.values) != 1:
        return state, deductions
    a_value = a_domain.values[0].name
    # Value-level dependencies make leak detection precise for this fixture.
    depends = (f"A={a_value}",)
    if a_value == "a1":
        target = next((v for v in c_domain.values if v.name == "c2"), None)
        if target is not None:
            deductions.append(
                CertifiedDeduction(
                    hole_id="C",
                    removed=target,
                    depends_on=depends,
                    certificate="cert-a1-not-c2",
                )
            )
            state = state.remove_value("C", target)
    elif a_value == "a2":
        target = next((v for v in c_domain.values if v.name == "c1"), None)
        if target is not None:
            deductions.append(
                CertifiedDeduction(
                    hole_id="C",
                    removed=target,
                    depends_on=depends,
                    certificate="cert-a2-not-c1",
                )
            )
            state = state.remove_value("C", target)
    return state, deductions


def _select_hole(state: FixtureState) -> str | None:
    """Smallest live domain first, then canonical hole id."""
    unresolved = [h for h in state.holes if len(h.values) > 1]
    if not unresolved:
        return None
    return min(unresolved, key=lambda h: (len(h.values), h.hole_id)).hole_id


def run_ablation_search(
    state: FixtureState,
    policy: AblationPolicy,
    *,
    ranker_order: tuple[str, ...] = ("a1", "a2"),
    max_decisions: int = 100,
    max_backtracks: int = 100,
) -> AblationResult:
    """Run one of the four ablation policies on the closed fixture.

    The implementation is intentionally self-contained so it can serve as a
    benchmark-only adapter without touching the production solver controller.
    """
    if policy not in AblationPolicy:
        raise ValueError(f"unknown ablation policy {policy!r}")

    def rank_values(values: tuple[FixtureValue, ...]) -> tuple[FixtureValue, ...]:
        order = {name: idx for idx, name in enumerate(ranker_order)}
        return tuple(sorted(values, key=lambda v: order.get(v.name, len(ranker_order))))

    def replace_hole_value(
        state: FixtureState, hole_id: str, value: FixtureValue
    ) -> FixtureState:
        """Return *state* with *hole_id* fixed to *value*, preserving reductions."""
        new_holes = tuple(
            FixtureHole(h.hole_id, (value,)) if h.hole_id == hole_id else h
            for h in state.holes
        )
        return FixtureState(
            holes=new_holes,
            decision_level=state.decision_level,
            parent_fingerprint=state.parent_fingerprint,
        )

    decisions: list[DecisionRecord] = []
    all_deductions: list[CertifiedDeduction] = []
    nogoods: list[NogoodRecord] = []
    stack: list[_Frame] = []
    current = state
    level = 0
    decision_counter = 0
    nodes = 0
    backtracks = 0
    restored_fingerprint: str | None = None
    stop_reason: str | None = None

    def record_nogood(hole_id: str, value: FixtureValue, provenance: str) -> None:
        nogoods.append(
            NogoodRecord(hole_id=hole_id, value=value, provenance=provenance)
        )

    def backtrack(provenance: str) -> bool:
        nonlocal current, level, backtracks, restored_fingerprint, decision_counter
        while stack:
            frame = stack[-1]
            backtracks += 1
            record_nogood(frame.hole_id, frame.chosen, provenance)
            if backtracks > max_backtracks:
                return False
            if policy is AblationPolicy.CERTIFIED_TRAIL:
                # Discard deductions that depended on the failed assumption.
                for d in frame.deductions:
                    all_deductions.remove(d)
            if frame.remaining:
                new_chosen = frame.remaining.pop(0)
                restored_fingerprint = frame.pre_state.fingerprint
                if policy is AblationPolicy.CERTIFIED_TRAIL:
                    # Restore the pre-decision checkpoint and commit the alternative.
                    current = frame.pre_state.with_decision(
                        frame.hole_id, new_chosen
                    )
                    level = frame.level + 1
                elif policy in (
                    AblationPolicy.MONOTONE_PROPOSAL,
                    AblationPolicy.PARTIAL_RETRACT,
                ):
                    # Keep the reductions produced under the failed assumption and
                    # switch to the next alternative.  This is intentionally unsafe.
                    current = replace_hole_value(
                        frame.post_state, frame.hole_id, new_chosen
                    )
                    level = frame.level + 1
                else:  # certified_only_no_branch never reaches here
                    current = frame.pre_state
                    level = frame.level

                decision_counter += 1
                decisions.append(
                    DecisionRecord(
                        decision_id=f"d{decision_counter}",
                        level=level,
                        hole_id=frame.hole_id,
                        chosen=new_chosen,
                        alternatives=tuple(frame.remaining),
                        before_fingerprint=frame.pre_state.fingerprint,
                        after_fingerprint=current.fingerprint,
                        policy=policy,
                    )
                )
                stack[-1] = _Frame(
                    pre_state=frame.pre_state,
                    post_state=current,
                    hole_id=frame.hole_id,
                    chosen=new_chosen,
                    remaining=frame.remaining,
                    level=frame.level,
                    deductions=(
                        []
                        if policy is AblationPolicy.CERTIFIED_TRAIL
                        else frame.deductions
                    ),
                )
                return True
            stack.pop()
        return False

    while True:
        if policy is AblationPolicy.CERTIFIED_ONLY_NO_BRANCH:
            current, closure_deductions = _apply_deductions(current)
            all_deductions.extend(closure_deductions)
            nodes += 1
            if current.is_bottom:
                stop_reason = "certified_bottom_no_branch"
                return _build_result(
                    policy,
                    "certified_unsat",
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            if current.is_fully_assigned:
                accepted, terminal = _check_terminal(current)
                status = "solved" if accepted else "unknown"
                stop_reason = None if accepted else "terminal_rejected_no_branch"
                return _build_result(
                    policy,
                    status,
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            stop_reason = "unresolved_no_branch"
            return _build_result(
                policy,
                "unknown",
                current,
                decisions,
                all_deductions,
                nogoods,
                backtracks,
                nodes,
                restored_fingerprint,
                stop_reason,
            )

        closed, closure_deductions = _apply_deductions(current)
        if stack:
            stack[-1].deductions = list(closure_deductions)
            stack[-1].post_state = closed
        all_deductions.extend(closure_deductions)
        nodes += 1

        if closed.is_bottom:
            if not backtrack("certified_bottom"):
                stop_reason = "exhausted"
                return _build_result(
                    policy,
                    "certified_unsat",
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            continue

        if closed.is_fully_assigned:
            accepted, terminal = _check_terminal(closed)
            if accepted:
                return _build_result(
                    policy,
                    "solved",
                    closed,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    None,
                    terminal=terminal,
                )
            if not backtrack("terminal_verifier_failure"):
                stop_reason = "terminal_rejected_exhausted"
                return _build_result(
                    policy,
                    "unknown",
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            continue

        if decision_counter >= max_decisions:
            stop_reason = "budget:max_decisions"
            return _build_result(
                policy,
                "budget_exhausted",
                current,
                decisions,
                all_deductions,
                nogoods,
                backtracks,
                nodes,
                restored_fingerprint,
                stop_reason,
            )

        hole_id = _select_hole(closed)
        if hole_id is None:
            if not backtrack("no_branching_hole"):
                stop_reason = "no_branching_hole"
                return _build_result(
                    policy,
                    "unknown",
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            continue

        live = tuple(
            v
            for v in closed.domain(hole_id).values
            if not any(
                n.hole_id == hole_id and n.value.name == v.name for n in nogoods
            )
        )
        if not live:
            if not backtrack("nogood_empty_domain"):
                stop_reason = "nogood_empty_domain"
                return _build_result(
                    policy,
                    "certified_unsat",
                    current,
                    decisions,
                    all_deductions,
                    nogoods,
                    backtracks,
                    nodes,
                    restored_fingerprint,
                    stop_reason,
                )
            continue

        ordered = rank_values(live)
        chosen, alternatives = ordered[0], ordered[1:]
        pre = closed
        after = closed.with_decision(hole_id, chosen)
        decision_counter += 1
        level += 1
        decisions.append(
            DecisionRecord(
                decision_id=f"d{decision_counter}",
                level=level,
                hole_id=hole_id,
                chosen=chosen,
                alternatives=alternatives,
                before_fingerprint=pre.fingerprint,
                after_fingerprint=after.fingerprint,
                policy=policy,
            )
        )
        stack.append(
            _Frame(
                pre_state=pre,
                post_state=after,
                hole_id=hole_id,
                chosen=chosen,
                remaining=list(alternatives),
                level=level - 1,
                deductions=[],
            )
        )
        current = after


def _is_dependency_active(
    deduction: CertifiedDeduction, state: FixtureState
) -> bool:
    """Return True if every dependency of *deduction* is active in *state*."""
    for dep in deduction.depends_on:
        if "=" in dep:
            hole_id, value_name = dep.split("=", 1)
            domain = state.domain(hole_id)
            if domain is None or value_name not in {v.name for v in domain.values}:
                return False
        else:
            domain = state.domain(dep)
            if domain is None or len(domain.values) != 1:
                return False
    return True


def _build_result(
    policy: AblationPolicy,
    status: str,
    state: FixtureState,
    decisions: list[DecisionRecord],
    deductions: list[CertifiedDeduction],
    nogoods: list[NogoodRecord],
    backtracks: int,
    nodes: int,
    restored_fingerprint: str | None,
    stop_reason: str | None,
    *,
    terminal: tuple[tuple[str, str], ...] = (),
) -> AblationResult:
    accepted = _accepted_terminals()
    if policy is AblationPolicy.CERTIFIED_ONLY_NO_BRANCH:
        # UNKNOWN is the expected safe outcome for the no-branch baseline.
        false_prune = False
        unknown_violation = False
    else:
        false_prune = (
            status in ("certified_unsat", "unknown", "budget_exhausted")
            and bool(accepted - set(terminal))
        )
        unknown_violation = status == "certified_unsat" and bool(accepted)
    leaked = tuple(d for d in deductions if not _is_dependency_active(d, state))
    return AblationResult(
        policy=policy,
        status=status,
        terminal=terminal,
        decisions=tuple(decisions),
        deductions=tuple(deductions),
        nogoods=tuple(nogoods),
        backtracks=backtracks,
        nodes=nodes,
        false_prune=false_prune,
        unknown_violation=unknown_violation,
        leaked_deductions=leaked,
        restored_fingerprint=restored_fingerprint,
        stop_reason=stop_reason,
    )


def build_ablation_fixture() -> FixtureState:
    """Return the deterministic start state for the ablation fixture."""
    return FixtureState(
        holes=(
            FixtureHole(
                "A",
                (FixtureValue("a1"), FixtureValue("a2")),
            ),
            FixtureHole(
                "C",
                (FixtureValue("c1"), FixtureValue("c2")),
            ),
        )
    )


def build_manifest(
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    include_unsafe: bool = True,
) -> Slm135Manifest:
    """Return the SLM-135 ablation manifest."""
    arms = [
        Slm135Arm("trail", AblationPolicy.CERTIFIED_TRAIL),
        Slm135Arm("certified_only", AblationPolicy.CERTIFIED_ONLY_NO_BRANCH),
    ]
    if include_unsafe:
        arms.extend(
            [
                Slm135Arm("monotone", AblationPolicy.MONOTONE_PROPOSAL),
                Slm135Arm("partial", AblationPolicy.PARTIAL_RETRACT),
            ]
        )
    return Slm135Manifest(
        arms=tuple(arms),
        seeds=seeds,
        status="not_run",
        claim_class="wiring",
    )


def _verdict_from_rows(rows: list[Slm135Row]) -> str:
    trail_rows = [r for r in rows if r.policy is AblationPolicy.CERTIFIED_TRAIL]
    unsafe_rows = [
        r
        for r in rows
        if r.policy in (AblationPolicy.MONOTONE_PROPOSAL, AblationPolicy.PARTIAL_RETRACT)
    ]
    if any(r.false_prune for r in trail_rows):
        return "production_trail_bug"
    if any(r.false_prune or r.leaked_deduction_count for r in unsafe_rows):
        return "trail_required"
    if all(r.status == "unknown" for r in rows if r.policy is AblationPolicy.CERTIFIED_ONLY_NO_BRANCH):
        return "certified_only_already_safe"
    return "dependency_tracking_required"


def run_fixture_matrix(
    manifest: Slm135Manifest,
    *,
    output_dir: Path | None = None,
    run_id: str = "slm135_fixture",
) -> Slm135Report:
    """Run the ablation fixture for every arm and seed."""
    rows: list[Slm135Row] = []
    start_state = build_ablation_fixture()
    for arm in manifest.arms:
        for seed in manifest.seeds:
            # Vary the ranker order by seed to exercise decision-order sensitivity.
            order = ("a1", "a2") if seed % 2 == 0 else ("a2", "a1")
            result = run_ablation_search(
                start_state,
                arm.policy,
                ranker_order=order,
            )
            notes = [
                f"ranker_order={order}",
                "fixture-only: hand-written closed CSP with assumption-dependent deductions",
            ]
            rows.append(
                Slm135Row(
                    arm_id=arm.arm_id,
                    policy=arm.policy,
                    seed=seed,
                    status=result.status,
                    result=result,
                    decision_count=len(result.decisions),
                    backtrack_count=result.backtracks,
                    false_prune=result.false_prune,
                    leaked_deduction_count=len(result.leaked_deductions),
                    notes=notes,
                )
            )

    verdict = _verdict_from_rows(rows)
    report = Slm135Report(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        verdict=verdict,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm135_trailed_assumptions_ablation",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm135_trailed_assumptions_report.json")
    return report


def render_markdown(report: Slm135Report) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-135 / EFS4-01: Trailed-assumptions ablation fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        f"Verdict: **{report.verdict}**",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Activation gate",
        "",
        report.manifest.activation_gate,
        "",
        "## Rows",
        "",
        "| Arm | Policy | Seed | Status | Decisions | Backtracks | False prune | Leaked deductions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.policy.value} | {row.seed} | "
            f"{row.status} | {row.decision_count} | {row.backtrack_count} | "
            f"{row.false_prune} | {row.leaked_deduction_count} |"
        )

    lines.extend(
        [
            "",
            "## Verdict interpretation",
            "",
            "* ``trail_required`` — unsafe controls exhibited a false prune or leaked "
            "deduction that ``certified_trail`` avoided.",
            "* ``certified_only_already_safe`` — the repository architecture never needs "
            "reversible decisions for this fixture.",
            "* ``dependency_tracking_required`` — explicit decision rollback is "
            "insufficient, but assumption-dependency retraction closes the gap.",
            "* ``production_trail_bug`` — ``certified_trail`` itself produced a false "
            "prune (regression).",
            "",
            "## Fixture caveat",
            "",
            "This is wiring-only evidence. The fixture is a hand-written two-hole CSP "
            "with one assumption-dependent deduction rule. It exercises the formal "
            "boundary between certified deductions and reversible decisions, but it "
            "is not a frontier-scale natural-recovery campaign and makes no ship-gate "
            "claim. Frontier/natural cases require the preregistered activation gate.",
            "",
        ]
    )
    return "\n".join(lines)
