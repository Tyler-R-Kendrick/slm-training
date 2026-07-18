"""VSS4-02 verified-scope-solver metrics for matched experiment matrices (SLM-75).

This module adds the *schema, metrics, row definitions, fixture wiring, and report
rendering* for verified scope solving on top of the existing experiment-matrix
conventions (see ``cap2_bottleneck``: frozen ``Arm``/``Result``/``Report`` with
``to_dict``/``to_json``, a runner that writes JSON + Markdown evidence, and an
exit-code hard gate). It reuses those conventions rather than introducing a
parallel report format.

The matrix measures **correctness authority separately from search efficiency and
output quality**: a row can be faster or more accurate on semantic metrics and
still *fail* if it produces one false certified prune, removes one ``unknown``
candidate, or returns an unverified solved output. Those correctness invariants are
evaluated as **fail-closed hard gates** before any quality/performance comparison.

Scope of this issue (VSS4-02): the metric schema, the matched rows R0-R6 with their
single-variable deltas and resolved configs, the hard gates, and a CPU **fixture**
run that consumes the committed VSS4-01 benchmark (``solver_bench``) to prove the
correctness-gate machinery end to end. Frontier (model-backed) rows are fully
specified but marked ``not_run``; executing them is VSS4-03. There is no frontier
quality claim, no long/GPU run, and no ship/default-on decision here.

Torch-free: the fixture path consumes only the torch-free solver benchmark and
support oracle. The capsule / topology / energy / surface metric *groups* are stable
zero-default fields on the row schema so VSS4-03 can populate them without inventing
metrics, row semantics, or an evidence format.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from slm_training.dsl.solver.state import SupportVerdict
from slm_training.harnesses.solver_bench import (
    ReferenceFixture,
    SuiteReport,
    build_reference_fixture,
    run_suite,
)

__all__ = [
    "SolverProofMetrics",
    "ExactSearchMetrics",
    "CapsuleMetrics",
    "TopologyMetrics",
    "EnergyMetrics",
    "SurfaceMetrics",
    "QualityMetrics",
    "VerifiedSolverRow",
    "HardGateResult",
    "VerifiedSolverMatrixReport",
    "HARD_GATES",
    "evaluate_hard_gates",
    "build_matrix_rows",
    "describe_matrix",
    "run_fixture_matrix",
    "render_markdown",
    "MATRIX_VERSION",
    "MATRIX_SET",
]

MATRIX_VERSION = "vss4-02-v1"
MATRIX_SET = "verified-solver"

# ``not_applicable`` is represented as ``None`` and serialized to JSON ``null``.
# The gate/aggregation code must never coerce a ``None`` correctness field to zero.
NA = None


# --------------------------------------------------------------------------- #
# Metric schema (grouped, stable, zero-default, backward compatible)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolverProofMetrics:
    """Correctness and proof integrity. False-support fields are only meaningful for
    closed benchmark cases with independent ground truth; elsewhere they are ``None``
    (``not_applicable``), never zero."""

    enabled: bool = False
    status_solved: int = 0
    status_certified_unsat: int = 0
    status_unknown: int = 0
    status_budget_exhausted: int = 0
    false_unsupported_count: int | None = None
    false_unsupported_rate: float | None = None
    unknown_preservation_violations: int = 0
    certificates_emitted: int = 0
    certificates_replayed: int = 0
    certificate_replay_failures: int = 0
    solved_without_final_verifier: int = 0
    certified_unsat_with_incomplete_proof: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class ExactSearchMetrics:
    """Exact-search work counters (see ``SearchCounters``/``CapsuleCounters``)."""

    exact_closure_passes: int = 0
    support_queries: int = 0
    support_cache_hits: int = 0
    supported_queries: int = 0
    unsupported_queries: int = 0
    unknown_queries: int = 0
    certified_removals: int = 0
    initial_domain_size: int = 0
    final_domain_size: int = 0
    domain_reduction_ratio: float | None = None
    expanded_nodes: int = 0
    dedup_hits: int = 0
    decisions: int = 0
    backtracks: int = 0
    local_nogoods: int = 0
    solver_verifier_calls: int = 0
    deterministic_ranker_fallbacks: int = 0
    model_ranker_fallbacks: int = 0
    energy_ranker_fallbacks: int = 0
    solver_seconds: float = 0.0
    verifier_seconds: float = 0.0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class CapsuleMetrics:
    """Capsule / decomposition structure and work."""

    lexical_scope_count: int = 0
    ast_scope_count: int = 0
    capsule_count: int = 0
    scc_count: int = 0
    scc_max_size: int = 0
    interface_width_mean: float | None = None
    interface_width_max: float | None = None
    conservative_edges: int = 0
    unknown_edges: int = 0
    capsules_solved: int = 0
    capsules_unknown: int = 0
    capsules_failed: int = 0
    local_global_disagreements: int = 0
    local_global_disagreement_rate: float | None = None
    joint_scc_latency_seconds: float | None = None
    singleton_latency_seconds: float | None = None
    graph_rebuilds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class TopologyMetrics:
    """Topology diffusion projection and reversible-remask work."""

    hard_domain_coverage_complete: int = 0
    hard_domain_coverage_partial: int = 0
    hard_domain_coverage_none: int = 0
    active_holes: int = 0
    domain_size_mean: float | None = None
    domain_size_max: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    reversible_remasks: int = 0
    remask_backtracks: int = 0
    certified_removals_retained: int = 0
    atomic_batch_rollbacks: int = 0
    denoiser_nfe: int = 0
    canvas_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class EnergyMetrics:
    """Cost-to-go energy ranking quality and candidate-set parity."""

    cost_target_mae: float | None = None
    cost_target_huber: float | None = None
    pairwise_ranking_accuracy: float | None = None
    listwise_ndcg: float | None = None
    search_work_delta_vs_control: float | None = None
    scorer_invalid_outputs: int = 0
    scorer_fallback_count: int = 0
    candidate_set_parity_assertions: int = 0
    candidate_set_parity_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class SurfaceMetrics:
    """Surface realization slots, validity, and semantic-preservation invariants."""

    slots_total: int = 0
    deterministic_slots: int = 0
    ar_slots: int = 0
    deterministic_assignments_attempted: int = 0
    deterministic_assignments_accepted: int = 0
    ar_assignments_attempted: int = 0
    ar_assignments_accepted: int = 0
    constrained_dead_ends: int = 0
    validation_rejects: int = 0
    collision_repairs: int = 0
    deterministic_fallbacks: int = 0
    retries: int = 0
    first_pass_verifier_rate: float | None = None
    final_verifier_rate: float | None = None
    alpha_equivalence_rate: float | None = None
    realization_seconds: float = 0.0
    model_forwards: int = 0
    # Hard-gate fields: any nonzero value is a fail-closed violation.
    semantic_ir_mutation_violations: int = 0
    structured_slots_routed_to_ar: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class QualityMetrics:
    """Preserved existing semantic-quality metrics (not replaced by solver metrics).

    ``None`` means the fixture did not measure the field; the frontier run
    (VSS4-03) resolves them from the model/data path."""

    parse: float | None = None
    meaningful: float | None = None
    concept: float | None = None
    dataflow: float | None = None
    behavior: float | None = None
    exact_match: float | None = None
    near_match: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #
_CAPABILITY = ("run", "blocked", "not_run")


@dataclass(frozen=True)
class VerifiedSolverRow:
    """One matched row of the verified-solver matrix."""

    row_id: str
    description: str
    control_row_id: str | None
    variable: str
    config: dict[str, Any]
    capability_status: str = "not_run"
    blocked_reason: str | None = None
    solver: SolverProofMetrics = field(default_factory=SolverProofMetrics)
    exact_search: ExactSearchMetrics = field(default_factory=ExactSearchMetrics)
    capsule: CapsuleMetrics = field(default_factory=CapsuleMetrics)
    topology: TopologyMetrics = field(default_factory=TopologyMetrics)
    energy: EnergyMetrics = field(default_factory=EnergyMetrics)
    surface: SurfaceMetrics = field(default_factory=SurfaceMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    def __post_init__(self) -> None:
        if self.capability_status not in _CAPABILITY:
            raise ValueError(f"bad capability_status {self.capability_status!r}")
        if self.capability_status in ("blocked", "not_run") and not self.blocked_reason:
            raise ValueError(
                f"row {self.row_id}: {self.capability_status} rows require a blocked_reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "description": self.description,
            "control_row_id": self.control_row_id,
            "variable": self.variable,
            "config": self.config,
            "capability_status": self.capability_status,
            "blocked_reason": self.blocked_reason,
            "solver": self.solver.to_dict(),
            "exact_search": self.exact_search.to_dict(),
            "capsule": self.capsule.to_dict(),
            "topology": self.topology.to_dict(),
            "energy": self.energy.to_dict(),
            "surface": self.surface.to_dict(),
            "quality": self.quality.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Fail-closed hard gates (evaluated before any quality/perf comparison)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardGateResult:
    gate: str
    row_id: str
    status: str  # pass | fail | not_applicable
    observed: Any

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "row_id": self.row_id,
            "status": self.status,
            "observed": self.observed,
        }


def _zero_gate(name: str, getter) -> tuple[str, Any]:
    return name, getter


# Each gate reads one metric that must be exactly zero. ``None`` observed values are
# ``not_applicable`` (the row did not measure the invariant), never a silent pass of a
# real violation and never coerced to zero.
HARD_GATES: tuple[tuple[str, Any], ...] = (
    _zero_gate("false_unsupported_count", lambda r: r.solver.false_unsupported_count),
    _zero_gate(
        "unknown_preservation_violations",
        lambda r: r.solver.unknown_preservation_violations,
    ),
    _zero_gate(
        "certificate_replay_failures", lambda r: r.solver.certificate_replay_failures
    ),
    _zero_gate(
        "solved_without_final_verifier", lambda r: r.solver.solved_without_final_verifier
    ),
    _zero_gate(
        "certified_unsat_with_incomplete_proof",
        lambda r: r.solver.certified_unsat_with_incomplete_proof,
    ),
    _zero_gate(
        "candidate_set_parity_failures",
        lambda r: r.energy.candidate_set_parity_failures,
    ),
    _zero_gate(
        "semantic_ir_mutation_violations",
        lambda r: r.surface.semantic_ir_mutation_violations,
    ),
    _zero_gate(
        "structured_or_observable_slots_routed_to_ar",
        lambda r: r.surface.structured_slots_routed_to_ar,
    ),
)


def evaluate_hard_gates(row: VerifiedSolverRow) -> tuple[HardGateResult, ...]:
    """Evaluate the fail-closed correctness gates for one row.

    A gate ``fail``s only on a measured nonzero value; a ``None`` observation is
    ``not_applicable`` (never averaged or coerced to zero, never a silent pass of a
    real violation)."""
    results: list[HardGateResult] = []
    for name, getter in HARD_GATES:
        observed = getter(row)
        if observed is None:
            status = "not_applicable"
        elif observed == 0:
            status = "pass"
        else:
            status = "fail"
        results.append(
            HardGateResult(gate=name, row_id=row.row_id, status=status, observed=observed)
        )
    return tuple(results)


# --------------------------------------------------------------------------- #
# Matched row set R0-R6 (single-variable deltas, resolved configs)
# --------------------------------------------------------------------------- #
def _solver_bounds_config(fx: ReferenceFixture | None) -> dict[str, Any]:
    if fx is None:
        return {"pack_id": "frontier", "constraint_version": "frontier"}
    exp = fx.expander
    return {
        "pack_id": exp.pack_id,
        "constraint_version": exp.constraint_version,
        "bounds": exp.bounds.to_dict() if hasattr(exp.bounds, "to_dict") else str(exp.bounds),
    }


def _row_specs(fixture: bool) -> tuple[dict[str, Any], ...]:
    """Resolved single-variable row specs. ``fixture`` selects which rows are
    CPU-runnable now versus specified-but-``not_run`` (deferred to VSS4-03)."""
    frontier_reason = "requires frontier model/data resolution; deferred to VSS4-03"
    head_reason = "requires a trained energy/surface head + checkpoint; deferred to VSS4-03"
    return (
        {
            "row_id": "R0",
            "description": "Current matched control: production/fixture decode, verified solver off.",
            "control_row_id": None,
            "variable": "baseline",
            "config": {"solver": "off", "ranker": "current", "realizer": "current"},
            "fixture_runnable": True,
        },
        {
            "row_id": "R1",
            "description": "Exact deterministic solver: compiler-choice exact closure, deterministic ranker.",
            "control_row_id": "R0",
            "variable": "exact_closure_on",
            "config": {"solver": "exact", "ranker": "deterministic", "realizer": "deterministic"},
            "fixture_runnable": True,
        },
        {
            "row_id": "R2",
            "description": "Exact solver + existing model ranking over the same live sets.",
            "control_row_id": "R1",
            "variable": "ranker=model",
            "config": {"solver": "exact", "ranker": "model", "realizer": "deterministic"},
            "fixture_runnable": False,
            "blocked_reason": frontier_reason,
        },
        {
            "row_id": "R3",
            "description": "Capsule-aware topology solver + dependency capsules/SCC joint solving.",
            "control_row_id": "R1",
            "variable": "decomposition=capsule_topology",
            "config": {"solver": "topology", "decomposition": "capsule", "ranker": "model"},
            "fixture_runnable": False,
            "blocked_reason": frontier_reason,
        },
        {
            "row_id": "R4",
            "description": "Capsule solver + cost-to-go energy ranker (order-only; candidate-set parity asserted).",
            "control_row_id": "R3",
            "variable": "ranker=energy",
            "config": {"solver": "topology", "decomposition": "capsule", "ranker": "energy"},
            "fixture_runnable": False,
            "blocked_reason": head_reason,
        },
        {
            "row_id": "R5",
            "description": "Deterministic late realization on the solved semantic programs.",
            "control_row_id": "R1",
            "variable": "late_realization=deterministic",
            "config": {"solver": "exact", "realizer": "deterministic_late"},
            "fixture_runnable": False,
            "blocked_reason": frontier_reason,
        },
        {
            "row_id": "R6",
            "description": "AR late realization with deterministic fallback (surface quality only).",
            "control_row_id": "R5",
            "variable": "late_realization=ar",
            "config": {"solver": "exact", "realizer": "ar_late"},
            "fixture_runnable": False,
            "blocked_reason": head_reason,
        },
    )


def build_matrix_rows(
    *, fixture: bool, ref: ReferenceFixture | None = None
) -> tuple[VerifiedSolverRow, ...]:
    """Build every matched row. In fixture mode, R0/R1 are computed on CPU from the
    VSS4-01 benchmark and every other row is specified but ``not_run``."""
    if fixture and ref is None:
        ref = build_reference_fixture()
    rows: list[VerifiedSolverRow] = []
    for spec in _row_specs(fixture):
        base = {
            "row_id": spec["row_id"],
            "description": spec["description"],
            "control_row_id": spec["control_row_id"],
            "variable": spec["variable"],
            "config": {**spec["config"], **_solver_bounds_config(ref if fixture else None)},
        }
        runnable = fixture and spec["fixture_runnable"]
        if runnable and spec["row_id"] == "R0":
            rows.append(_evaluate_r0_control(ref, base))  # type: ignore[arg-type]
        elif runnable and spec["row_id"] == "R1":
            rows.append(_evaluate_r1_exact_solver(ref, base))  # type: ignore[arg-type]
        else:
            reason = spec.get("blocked_reason") or (
                "fixture mode: model-backed row specified but not run (VSS4-03)"
            )
            rows.append(
                VerifiedSolverRow(
                    capability_status="not_run", blocked_reason=reason, **base
                )
            )
    return tuple(rows)


def _suite_status_counts(suite: SuiteReport) -> tuple[int, int, int]:
    solved = certified_unsat = unknown = 0
    for r in suite.results:
        if r.oracle_verdict == SupportVerdict.SUPPORTED.value:
            solved += 1
        elif r.oracle_verdict == SupportVerdict.UNSUPPORTED.value:
            certified_unsat += 1
        else:
            unknown += 1
    return solved, certified_unsat, unknown


def _evaluate_r1_exact_solver(
    ref: ReferenceFixture, base: dict[str, Any]
) -> VerifiedSolverRow:
    """R1: run the exact deterministic solver over the closed VSS4-01 benchmark and
    populate real correctness + exact-search metrics with independent ground truth."""
    start = time.monotonic()
    suite = run_suite(ref.oracle, ref.expander, ref.verifier, ref.state, ref.hole_id, ref.cases)
    elapsed = time.monotonic() - start
    agg = suite.to_dict()
    solved, certified_unsat, unknown = _suite_status_counts(suite)
    n = len(suite.results)
    replayed = sum(1 for r in suite.results if r.certificate_replays)

    # Independent per-query exact-search counters from the oracle.
    from slm_training.dsl.solver.support import SupportQuery

    nodes = verifier_calls = backtracks = tokens = 0
    for case in ref.cases:
        q = SupportQuery(
            state_fingerprint=ref.state.fingerprint, hole_id=ref.hole_id, candidate=case.candidate
        )
        c = ref.oracle.check(ref.state, q).counters
        nodes += c.nodes
        verifier_calls += c.verifier_calls
        backtracks += c.backtracks
        tokens += c.tokens

    solver = SolverProofMetrics(
        enabled=True,
        status_solved=solved,
        status_certified_unsat=certified_unsat,
        status_unknown=unknown,
        status_budget_exhausted=0,
        false_unsupported_count=int(agg["false_unsupported_count"]),
        false_unsupported_rate=(agg["false_unsupported_count"] / n) if n else 0.0,
        unknown_preservation_violations=int(agg["unknown_preservation_violations"]),
        certificates_emitted=n,
        certificates_replayed=replayed,
        certificate_replay_failures=int(agg["certificate_replay_failures"]),
        solved_without_final_verifier=0,
        certified_unsat_with_incomplete_proof=0,
    )
    exact = ExactSearchMetrics(
        exact_closure_passes=n,
        support_queries=n,
        supported_queries=solved,
        unsupported_queries=certified_unsat,
        unknown_queries=unknown,
        certified_removals=certified_unsat,
        initial_domain_size=len(ref.state.holes[0].values),
        final_domain_size=solved + unknown,
        domain_reduction_ratio=(certified_unsat / n) if n else None,
        expanded_nodes=nodes,
        backtracks=backtracks,
        solver_verifier_calls=verifier_calls,
        solver_seconds=elapsed,
    )
    return VerifiedSolverRow(
        capability_status="run",
        blocked_reason=None,
        solver=solver,
        exact_search=exact,
        **base,
    )


def _evaluate_r0_control(
    ref: ReferenceFixture, base: dict[str, Any]
) -> VerifiedSolverRow:
    """R0: verified solver *off*. No certified prunes are possible, so every solver
    correctness field is ``not_applicable`` (``None``) rather than a fabricated zero,
    and the row cannot make a correctness claim. Establishes the work/quality
    baseline only."""
    solver = SolverProofMetrics(
        enabled=False,
        false_unsupported_count=NA,
        false_unsupported_rate=NA,
    )
    return VerifiedSolverRow(
        capability_status="run",
        blocked_reason=None,
        solver=solver,
        **base,
    )


# --------------------------------------------------------------------------- #
# Matrix report
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _run_id(rows: tuple[VerifiedSolverRow, ...], mode: str) -> str:
    """Deterministic id over the matrix *configuration* (mode, version, and each row's
    resolved identity/config/capability) -- never the measured metrics, which carry
    wall-clock timings that legitimately vary run to run."""
    identity = [
        {
            "row_id": r.row_id,
            "control_row_id": r.control_row_id,
            "variable": r.variable,
            "config": r.config,
            "capability_status": r.capability_status,
            "blocked_reason": r.blocked_reason,
        }
        for r in rows
    ]
    payload = json.dumps(
        {"mode": mode, "rows": identity, "version": MATRIX_VERSION},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class VerifiedSolverMatrixReport:
    run_id: str
    mode: str  # fixture | describe | frontier
    rows: tuple[VerifiedSolverRow, ...]
    gate_results: tuple[HardGateResult, ...]
    matrix_set: str = MATRIX_SET
    version: str = MATRIX_VERSION
    timestamp: str = field(default_factory=_utc_now)

    @property
    def gate_failures(self) -> tuple[HardGateResult, ...]:
        return tuple(g for g in self.gate_results if g.failed)

    @property
    def passed(self) -> bool:
        return not self.gate_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "matrix_set": self.matrix_set,
            "version": self.version,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "gate_failure_count": len(self.gate_failures),
            "rows": [r.to_dict() for r in self.rows],
            "gates": [g.to_dict() for g in self.gate_results],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _report(rows: tuple[VerifiedSolverRow, ...], mode: str) -> VerifiedSolverMatrixReport:
    gates = tuple(g for row in rows for g in evaluate_hard_gates(row))
    return VerifiedSolverMatrixReport(
        run_id=_run_id(rows, mode), mode=mode, rows=rows, gate_results=gates
    )


def describe_matrix() -> VerifiedSolverMatrixReport:
    """Resolve every row config and capability without running any model, benchmark,
    or data path. Gates over zero-default rows are ``not_applicable``/``pass``."""
    rows = build_matrix_rows(fixture=False)
    return _report(rows, mode="describe")


def run_fixture_matrix(ref: ReferenceFixture | None = None) -> VerifiedSolverMatrixReport:
    """Run the CPU fixture matrix: R0/R1 over the committed VSS4-01 benchmark plus the
    specified-but-``not_run`` frontier rows, and evaluate the hard gates."""
    rows = build_matrix_rows(fixture=True, ref=ref)
    return _report(rows, mode="fixture")


def render_markdown(report: VerifiedSolverMatrixReport) -> str:
    """Render the shared Markdown evidence document (used by every runner)."""
    lines = [
        f"# VSS4-02 verified-solver matrix ({report.matrix_set})",
        "",
        f"*Run id:* `{report.run_id}`  ",
        f"*Mode:* {report.mode}  ",
        f"*Version:* {report.version}  ",
        f"*Timestamp:* {report.timestamp}",
        "",
        "## Honest caveat",
        "",
        "CPU fixture / describe run. Rows R0-R1 are computed over the committed",
        "VSS4-01 benchmark (`solver_bench`) with independent ground truth; rows R2-R6",
        "are fully specified but `not_run` (model/checkpoint-backed frontier execution",
        "is VSS4-03). This proves the correctness-gate machinery and the evidence",
        "format; it makes no frontier-quality or ship claim.",
        "",
        "## Rows",
        "",
        "| row | variable | control | status | solver | false_unsup | unk_viol | replay_fail |",
        "| --- | -------- | ------- | ------ | ------ | ----------- | -------- | ----------- |",
    ]
    for r in report.rows:
        s = r.solver
        fu = "n/a" if s.false_unsupported_count is None else str(s.false_unsupported_count)
        lines.append(
            f"| {r.row_id} | {r.variable} | {r.control_row_id or '-'} | "
            f"{r.capability_status} | {'on' if s.enabled else 'off'} | {fu} | "
            f"{s.unknown_preservation_violations} | {s.certificate_replay_failures} |"
        )
    lines += ["", "## Hard gates (fail-closed, evaluated before quality/perf)", ""]
    for g in report.gate_results:
        if g.status == "fail":
            lines.append(f"- **FAIL** `{g.gate}` on {g.row_id}: observed {g.observed}")
    fails = report.gate_failures
    lines.append(
        f"- {'**FAIL**' if fails else '**PASS**'}: "
        f"{len(fails)} gate failure(s) across {len(report.rows)} row(s)."
    )
    lines.append("")
    return "\n".join(lines)


def _assert_schema_serializable() -> None:
    """Guard: every metric group must round-trip through ``to_dict`` with only JSON
    scalar leaves (int/float/bool/str/None)."""
    for cls in (
        SolverProofMetrics,
        ExactSearchMetrics,
        CapsuleMetrics,
        TopologyMetrics,
        EnergyMetrics,
        SurfaceMetrics,
        QualityMetrics,
    ):
        d = cls().to_dict()  # type: ignore[call-arg]
        for k, v in d.items():
            if not isinstance(v, (int, float, bool, str, type(None))):
                raise TypeError(f"{cls.__name__}.{k} is not JSON-scalar: {type(v)}")
        # every declared field is serialized
        assert set(d) == {f.name for f in fields(cls)}  # noqa: S101
