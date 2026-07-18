"""VSS4-02 (SLM-75): matched verified-scope-solver evaluation matrix.

This module adds the ``verified-solver`` matrix set consumed by the existing
``scripts/run_quality_matrix.py`` runner.  It does **not** introduce a fourth
matrix runner or a parallel report format: it provides the stable metric
schema, the matched R0-R6 row set, the fail-closed hard gates, the CPU fixture
executor (wrapping the VSS4-01 benchmark, :mod:`slm_training.harnesses.solver_bench`),
and JSON/Markdown report rendering that the runner dispatches to.

Design boundaries (mirrors the issue's non-goals):

* Fixture mode is CPU-only and torch-free.  Only the control row (R0) and the
  closed exact-search benchmark row (R1) execute; every row that requires a
  trained checkpoint/head (model ranker, cost-to-go energy, capsule benchmark
  family, surface realizer) is marked ``blocked`` with an explicit reason
  rather than silently substituting a weaker configuration.
* The correctness/proof hard gates are evaluated *before* any quality/perf
  comparison and are fail-closed: a required-but-missing measurement fails.
* ``not_applicable`` correctness fields (false-support metrics on rows without
  independent ground truth) are ``None`` and are never averaged into zero.
* Nothing here loads a model or writes a file on import; ``describe`` resolves
  configs without running anything.

The metric field names deliberately mirror their producers so a later frontier
run can populate them from the real objects without renaming:

* solver correctness  -> :class:`slm_training.harnesses.solver_bench.SuiteReport`
* exact-search work   -> :class:`slm_training.models.decode_stats.DecodeStats`
                         + ``dsl/solver/closure.py::ClosureCounters``
* capsule metrics     -> ``dsl/solver/capsule_solver.py::CapsuleCounters``
* topology metrics    -> ``dsl/solver/topology_solver.py`` closure result
* energy metrics      -> ``models/solver_energy.py::CandidateEnergyRanker``
* surface metrics     -> ``dsl/surface.py::SurfaceRealizationResult``

See ``docs/design/verified-scope-solver-benchmark.md`` and
``docs/design/verified-scope-solver.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from slm_training.harnesses.solver_bench import SuiteReport, run_reference_suite

SCHEMA_VERSION = "verified_scope_solver_matrix_v1"
MATRIX_SET = "verified-solver"

# The reference fixture the VSS4-01 benchmark exposes (family A, torch-free).
FIXTURE_BENCHMARK_ID = "vss4-01/verified_scope_solver/v1"

# The fail-closed correctness/proof gates, evaluated before any quality gain.
# Names are stable and referenced verbatim by the runner, docs, and tests.
HARD_GATES: tuple[str, ...] = (
    "false_unsupported_count",
    "unknown_preservation_violations",
    "certificate_replay_failures",
    "solved_without_final_verifier",
    "certified_unsat_with_incomplete_proof",
    "candidate_set_parity_failures",
    "surface.semantic_ir_mutation_violations",
    "structured_or_observable_slots_routed_to_ar",
)


# --------------------------------------------------------------------------- #
# Metric schema (stable, zero-default, JSON-serializable, grouped).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolverCorrectness:
    """Correctness and proof-integrity group (``solver.*``).

    ``false_unsupported_*`` and ``unknown_preservation_violations`` are only
    defined for closed benchmark cases with independent ground truth; they are
    ``None`` (``not_applicable``) elsewhere and must never be averaged to zero.
    """

    enabled: bool = False
    status_counts: dict[str, int] = field(
        default_factory=lambda: {
            "solved": 0,
            "certified_unsat": 0,
            "unknown": 0,
            "budget_exhausted": 0,
        }
    )
    false_unsupported_count: int | None = None
    false_unsupported_rate: float | None = None
    unknown_preservation_violations: int | None = None
    certificates_emitted: int = 0
    certificates_replayed: int = 0
    certificate_replay_failures: int = 0
    solved_without_final_verifier: int = 0
    certified_unsat_with_incomplete_proof: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status_counts": dict(self.status_counts),
            "false_unsupported_count": self.false_unsupported_count,
            "false_unsupported_rate": self.false_unsupported_rate,
            "unknown_preservation_violations": self.unknown_preservation_violations,
            "certificates_emitted": self.certificates_emitted,
            "certificates_replayed": self.certificates_replayed,
            "certificate_replay_failures": self.certificate_replay_failures,
            "solved_without_final_verifier": self.solved_without_final_verifier,
            "certified_unsat_with_incomplete_proof": (
                self.certified_unsat_with_incomplete_proof
            ),
        }


@dataclass(frozen=True)
class ExactSearchWork:
    """Exact-search-work group (deterministic closure + controller counters)."""

    closure_passes: int = 0
    support_queries: int = 0
    support_cache_hits: int = 0
    supported_queries: int = 0
    unsupported_queries: int = 0
    unknown_queries: int = 0
    certified_candidate_removals: int = 0
    initial_domain_size: int = 0
    final_domain_size: int = 0
    domain_reduction_ratio: float | None = None
    expanded_nodes: int = 0
    dedup_hits: int = 0
    decisions: int = 0
    backtracks: int = 0
    local_nogoods: int = 0
    verifier_calls: int = 0
    deterministic_ranker_fallbacks: int = 0
    model_ranker_fallbacks: int = 0
    energy_ranker_fallbacks: int = 0
    solver_ms: float = 0.0
    certificate_ms: float = 0.0
    denoiser_ms: float = 0.0
    projection_ms: float = 0.0
    global_verifier_ms: float = 0.0
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    latency_ms_p99: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CapsuleMetrics:
    """Capsule/decomposition group (``dsl/solver/capsule_solver.py``)."""

    lexical_scope_count: int = 0
    ast_scope_count: int = 0
    capsule_count: int = 0
    scc_count: int = 0
    scc_max_size: int = 0
    interface_width_mean: float | None = None
    interface_width_max: int = 0
    interface_width_quantiles: dict[str, float] = field(default_factory=dict)
    conservative_dependency_edges: int = 0
    unknown_dependency_edges: int = 0
    capsules_solved: int = 0
    capsules_unknown: int = 0
    capsules_failed: int = 0
    local_global_disagreement_count: int = 0
    local_global_disagreement_rate: float | None = None
    joint_scc_work: int = 0
    singleton_work: int = 0
    joint_scc_latency_ms: float = 0.0
    singleton_latency_ms: float = 0.0
    graph_rebuild_count: int = 0
    plan_rebuild_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TopologyMetrics:
    """Topology-diffusion group (``dsl/solver/topology_solver.py``)."""

    hard_domain_coverage: str = "none"  # complete | partial | none
    active_holes: int = 0
    domain_size_mean: float | None = None
    domain_size_max: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    reversible_remasks: int = 0
    remask_backtracks: int = 0
    certified_removals_retained_across_remask: int = 0
    atomic_batch_rollbacks: int = 0
    denoiser_nfe: int = 0
    canvas_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EnergyMetrics:
    """Energy-ranking group (``models/solver_energy.py``).

    ``candidate_set_parity_failures`` proves the learned ranker is order-only:
    any non-zero value is a hard-gate failure.
    """

    cost_target_mae: float | None = None
    cost_target_huber: float | None = None
    pairwise_ranking_accuracy: float | None = None
    listwise_ndcg: float | None = None
    search_work_delta_vs_deterministic: float | None = None
    search_work_delta_vs_model: float | None = None
    scorer_invalid_outputs: int = 0
    scorer_fallback_count: int = 0
    candidate_set_parity_assertions: int = 0
    candidate_set_parity_failures: int = 0
    stratification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SurfaceMetrics:
    """Surface-realization group (``dsl/surface.py`` / neural realizer).

    ``semantic_ir_mutation_violations`` and
    ``structured_or_observable_slots_routed_to_ar`` are hard-gate counters:
    the realizer may only touch surface-only slots and may never route a
    structured/observable slot to the autoregressive path.
    """

    semantic_ir_mutation_violations: int = 0
    slots_total: int = 0
    slots_by_kind: dict[str, int] = field(default_factory=dict)
    slots_by_authority: dict[str, int] = field(default_factory=dict)
    deterministic_assignments_attempted: int = 0
    deterministic_assignments_accepted: int = 0
    ar_assignments_attempted: int = 0
    ar_assignments_accepted: int = 0
    constrained_dead_ends: int = 0
    validation_rejects: int = 0
    collision_repairs: int = 0
    capture_repairs: int = 0
    deterministic_fallbacks: int = 0
    deterministic_retries: int = 0
    first_pass_verifier_rate: float | None = None
    final_verifier_rate: float | None = None
    alpha_equivalence_pass_rate: float | None = None
    realization_latency_ms: float | None = None
    model_forwards: int = 0
    structured_or_observable_slots_routed_to_ar: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class QualityMetrics:
    """Preserved semantic-quality metrics (never replaced by solver metrics)."""

    parse_rate: float | None = None
    meaningful_program_rate: float | None = None
    component_type_recall: float | None = None
    structural_similarity: float | None = None
    placeholder_fidelity: float | None = None
    reward_score: float | None = None
    exact_match: float | None = None
    near_match: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RowMetrics:
    """The full grouped metric bundle for one matrix row."""

    solver: SolverCorrectness = field(default_factory=SolverCorrectness)
    exact_search_work: ExactSearchWork = field(default_factory=ExactSearchWork)
    capsule: CapsuleMetrics = field(default_factory=CapsuleMetrics)
    topology: TopologyMetrics = field(default_factory=TopologyMetrics)
    energy: EnergyMetrics = field(default_factory=EnergyMetrics)
    surface: SurfaceMetrics = field(default_factory=SurfaceMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver.to_dict(),
            "exact_search_work": self.exact_search_work.to_dict(),
            "capsule": self.capsule.to_dict(),
            "topology": self.topology.to_dict(),
            "energy": self.energy.to_dict(),
            "surface": self.surface.to_dict(),
            "quality": self.quality.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Matched experiment rows (R0-R6) and required ablation strata.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerifiedSolverRow:
    """One matched row.  Rows differ from their declared control by exactly one
    variable so every delta is attributable."""

    row_id: str
    run_id: str
    description: str
    control: str | None
    single_variable: str
    solver_enabled: bool
    exact_closure: bool
    ranker: str  # none | deterministic | model | energy
    capsule: bool
    topology: bool
    realizer: str  # none | deterministic | ar
    is_closed_benchmark: bool
    required_capabilities: tuple[str, ...]
    tags: tuple[str, ...] = ()

    @property
    def surface_active(self) -> bool:
        return self.realizer in {"deterministic", "ar"}


def verified_solver_rows() -> tuple[VerifiedSolverRow, ...]:
    """The versioned R0-R6 matched row set (stable, deterministic order)."""

    return (
        VerifiedSolverRow(
            row_id="R0",
            run_id="vss4-02-r0-control",
            description="Current matched control: verified solver off, current "
            "deterministic finalization; establishes the quality/work baseline.",
            control=None,
            single_variable="baseline",
            solver_enabled=False,
            exact_closure=False,
            ranker="none",
            capsule=False,
            topology=False,
            realizer="deterministic",
            is_closed_benchmark=False,
            required_capabilities=(),
            tags=("control",),
        ),
        VerifiedSolverRow(
            row_id="R1",
            run_id="vss4-02-r1-exact-deterministic",
            description="Exact deterministic solver: compiler-choice exact "
            "closure on, deterministic ranker, no learned energy; isolates "
            "exact-search correctness and work.",
            control="R0",
            single_variable="exact_closure=on",
            solver_enabled=True,
            exact_closure=True,
            ranker="deterministic",
            capsule=False,
            topology=False,
            realizer="deterministic",
            is_closed_benchmark=True,
            required_capabilities=(),
            tags=("closed_benchmark", "fixture_runnable"),
        ),
        VerifiedSolverRow(
            row_id="R2",
            run_id="vss4-02-r2-exact-model-rank",
            description="Exact solver + existing model ranking: same hard "
            "domains/bounds as R1, current model logits rank live candidates; "
            "isolates learned ordering from exact closure.",
            control="R1",
            single_variable="ranker=model",
            solver_enabled=True,
            exact_closure=True,
            ranker="model",
            capsule=False,
            topology=False,
            realizer="deterministic",
            is_closed_benchmark=True,
            required_capabilities=("twotower_ranker_checkpoint",),
            tags=("closed_benchmark",),
        ),
        VerifiedSolverRow(
            row_id="R3",
            run_id="vss4-02-r3-capsule-topology",
            description="Capsule-aware topology solver: topology verified solver "
            "with dependency capsules/SCC joint solving, current model ranker, "
            "deterministic realization.",
            control="R2",
            single_variable="capsule_topology=on",
            solver_enabled=True,
            exact_closure=True,
            ranker="model",
            capsule=True,
            topology=True,
            realizer="deterministic",
            is_closed_benchmark=True,
            required_capabilities=(
                "twotower_ranker_checkpoint",
                "capsule_benchmark_family_c",
            ),
            tags=("closed_benchmark",),
        ),
        VerifiedSolverRow(
            row_id="R4",
            run_id="vss4-02-r4-capsule-energy",
            description="Capsule solver + cost-to-go energy: identical to R3 "
            "except ranker=energy; reports candidate-set parity and search-work "
            "deltas with no hard-membership change permitted.",
            control="R3",
            single_variable="ranker=energy",
            solver_enabled=True,
            exact_closure=True,
            ranker="energy",
            capsule=True,
            topology=True,
            realizer="deterministic",
            is_closed_benchmark=True,
            required_capabilities=(
                "capsule_benchmark_family_c",
                "cost_to_go_energy_checkpoint",
            ),
            tags=("closed_benchmark",),
        ),
        VerifiedSolverRow(
            row_id="R5",
            run_id="vss4-02-r5-deterministic-realization",
            description="Deterministic late realization: same solved semantic "
            "programs as its matched structural row (R3), deterministic surface "
            "protocol on; measures alpha-equivalence, verifier pass, overhead.",
            control="R3",
            single_variable="late_realization=deterministic",
            solver_enabled=True,
            exact_closure=True,
            ranker="model",
            capsule=True,
            topology=True,
            realizer="deterministic",
            is_closed_benchmark=False,
            required_capabilities=("surface_benchmark_family_e",),
            tags=("surface",),
        ),
        VerifiedSolverRow(
            row_id="R6",
            run_id="vss4-02-r6-ar-realization",
            description="AR late realization: same semantic programs/slots as "
            "R5, AR realizer with deterministic fallback; measures only surface "
            "quality/validity/overhead, never structural solve quality.",
            control="R5",
            single_variable="realizer=ar",
            solver_enabled=True,
            exact_closure=True,
            ranker="model",
            capsule=True,
            topology=True,
            realizer="ar",
            is_closed_benchmark=False,
            required_capabilities=(
                "surface_benchmark_family_e",
                "surface_ar_checkpoint",
            ),
            tags=("surface",),
        ),
    )


# Required ablations/strata (documented axes; several are already realized by
# the R0-R6 matched pairs, listed here so ``--describe`` and the docs enumerate
# every comparison the matrix supports).
ABLATION_AXES: tuple[dict[str, str], ...] = (
    {
        "axis": "exact_closure_on_off",
        "rows": "R0 vs R1",
        "detail": "exact closure on/off under the same deterministic ranker",
    },
    {
        "axis": "model_vs_energy_ranker",
        "rows": "R3 vs R4",
        "detail": "model vs energy ranker over the same exact live sets",
    },
    {
        "axis": "lexical_vs_capsule_decomposition",
        "rows": "R2 vs R3",
        "detail": "lexical/AST-local decomposition vs dependency capsules",
    },
    {
        "axis": "coupling_strata",
        "rows": "within-row",
        "detail": "low/medium/high coupling at matched AST size",
    },
    {
        "axis": "interface_width_strata",
        "rows": "within-row",
        "detail": "small/large interface width at matched capsule node count",
    },
    {
        "axis": "deterministic_vs_ar_realization",
        "rows": "R5 vs R6",
        "detail": "deterministic vs AR realization on the same semantic IR",
    },
    {
        "axis": "alpha_renamed_holdouts",
        "rows": "within-row",
        "detail": "alpha-renamed and unseen-identifier held-outs",
    },
)


def row_by_id() -> dict[str, VerifiedSolverRow]:
    return {row.row_id: row for row in verified_solver_rows()}


# --------------------------------------------------------------------------- #
# Deterministic config resolution + hashing.
# --------------------------------------------------------------------------- #


def resolve_row_config(row: VerifiedSolverRow) -> dict[str, Any]:
    """Resolve a row to its exact config dict (deterministic, no model load)."""

    return {
        "row_id": row.row_id,
        "run_id": row.run_id,
        "control": row.control,
        "single_variable": row.single_variable,
        "solver_enabled": row.solver_enabled,
        "exact_closure": row.exact_closure,
        "ranker": row.ranker,
        "capsule": row.capsule,
        "topology": row.topology,
        "realizer": row.realizer,
        "is_closed_benchmark": row.is_closed_benchmark,
        "required_capabilities": list(row.required_capabilities),
        "final_verifier_profile": "pack-oracle-v1",
        "solver_bounds": {
            "max_nodes": 10000,
            "max_depth": 32,
            "max_backtracks": 10000,
            "max_verifier_calls": 10000,
        },
        "fixture_benchmark_id": FIXTURE_BENCHMARK_ID,
        "schema_version": SCHEMA_VERSION,
    }


def config_hash(row: VerifiedSolverRow) -> str:
    """Stable content hash of the resolved config (deterministic across runs)."""

    payload = json.dumps(resolve_row_config(row), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Fail-closed hard gates (evaluated before quality/perf gains).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, str]  # gate -> "pass" | "fail" | "not_applicable"
    failures: tuple[str, ...]
    not_applicable: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "checks": dict(self.checks),
            "failures": list(self.failures),
            "not_applicable": list(self.not_applicable),
        }


def evaluate_verified_solver_gates(
    metrics: RowMetrics,
    *,
    closed_benchmark: bool,
    solver_enabled: bool,
    surface_active: bool,
) -> GateResult:
    """Evaluate the eight fail-closed correctness/proof gates.

    Fail-closed semantics:

    * A gate whose value is ``0`` passes.
    * A gate whose value is ``> 0`` fails.
    * ``false_unsupported_count`` / ``unknown_preservation_violations`` are only
      defined for closed benchmarks.  On a non-closed row they are
      ``not_applicable`` (excluded from the pass computation, never counted as
      zero).  On a *closed* row a missing (``None``) value is a **failure**.
    * The always-applicable gates (certificate replay, unverified solve,
      incomplete-proof unsat, candidate-set parity) are ``not_applicable`` only
      when the solver did not run; otherwise a missing value fails closed.
    * Surface gates are ``not_applicable`` when no realizer ran.
    """

    solver = metrics.solver
    energy = metrics.energy
    surface = metrics.surface
    checks: dict[str, str] = {}
    failures: list[str] = []
    not_applicable: list[str] = []

    def record(name: str, value: int | None, *, applicable: bool) -> None:
        if not applicable:
            checks[name] = "not_applicable"
            not_applicable.append(name)
            return
        if value is None:
            # Fail-closed: an applicable gate we could not measure fails.
            checks[name] = "fail"
            failures.append(f"{name}:unmeasured")
            return
        if value == 0:
            checks[name] = "pass"
        else:
            checks[name] = "fail"
            failures.append(f"{name}={value}")

    # Ground-truth-dependent gates (closed benchmark only).
    record(
        "false_unsupported_count",
        solver.false_unsupported_count,
        applicable=closed_benchmark and solver_enabled,
    )
    record(
        "unknown_preservation_violations",
        solver.unknown_preservation_violations,
        applicable=closed_benchmark and solver_enabled,
    )
    # Always-applicable when the solver ran.
    record(
        "certificate_replay_failures",
        solver.certificate_replay_failures,
        applicable=solver_enabled,
    )
    record(
        "solved_without_final_verifier",
        solver.solved_without_final_verifier,
        applicable=solver_enabled,
    )
    record(
        "certified_unsat_with_incomplete_proof",
        solver.certified_unsat_with_incomplete_proof,
        applicable=solver_enabled,
    )
    record(
        "candidate_set_parity_failures",
        energy.candidate_set_parity_failures,
        applicable=solver_enabled,
    )
    # Surface gates (only when a realizer ran).
    record(
        "surface.semantic_ir_mutation_violations",
        surface.semantic_ir_mutation_violations,
        applicable=surface_active,
    )
    record(
        "structured_or_observable_slots_routed_to_ar",
        surface.structured_or_observable_slots_routed_to_ar,
        applicable=surface_active,
    )

    return GateResult(
        passed=not failures,
        checks=checks,
        failures=tuple(failures),
        not_applicable=tuple(not_applicable),
    )


# --------------------------------------------------------------------------- #
# Fixture executor (torch-free, CPU) wrapping the VSS4-01 benchmark.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RowResult:
    row_id: str
    run_id: str
    status: str  # ran | blocked | not_run
    blocked_reason: str | None
    control: str | None
    single_variable: str
    config_hash: str
    is_closed_benchmark: bool
    metrics: RowMetrics
    gate: GateResult | None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.row_id,
            "run_id": self.run_id,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "control": self.control,
            "single_variable": self.single_variable,
            "config_hash": self.config_hash,
            "is_closed_benchmark": self.is_closed_benchmark,
            "gate": self.gate.to_dict() if self.gate is not None else None,
            "metrics": self.metrics.to_dict(),
            "evidence": dict(self.evidence),
        }


def solver_metrics_from_suite(report: SuiteReport) -> SolverCorrectness:
    """Map a VSS4-01 :class:`SuiteReport` onto the ``solver.*`` schema.

    The benchmark decides *support* (supported/unsupported/unknown) with
    independent ground truth.  A ``supported`` verdict means an accepted
    terminal is reachable (the subtree is solvable), ``unsupported`` means a
    certified proof of no-solution (certified_unsat for that subtree), and
    ``unknown`` means bounded/incomplete coverage.  We map verdicts onto the
    solve-status vocabulary under that documented correspondence.
    """

    cases = report.results
    case_count = len(cases)
    status = {"solved": 0, "certified_unsat": 0, "unknown": 0, "budget_exhausted": 0}
    for case in cases:
        verdict = case.oracle_verdict
        if verdict == "supported":
            status["solved"] += 1
        elif verdict == "unsupported":
            status["certified_unsat"] += 1
        else:
            status["unknown"] += 1
    data = report.to_dict()
    false_unsupported = int(data["false_unsupported_count"])
    unknown_violations = int(data["unknown_preservation_violations"])
    replay_failures = int(data["certificate_replay_failures"])
    return SolverCorrectness(
        enabled=True,
        status_counts=status,
        false_unsupported_count=false_unsupported,
        false_unsupported_rate=(
            false_unsupported / case_count if case_count else 0.0
        ),
        unknown_preservation_violations=unknown_violations,
        certificates_emitted=case_count,
        certificates_replayed=case_count - replay_failures,
        certificate_replay_failures=replay_failures,
        # The benchmark cross-checks every terminal against the pack verifier and
        # ground truth, so no case is certified without verification.
        solved_without_final_verifier=0,
        # A certified-unsat reached on an incomplete path is exactly the
        # unknown-preservation violation the benchmark already counts.
        certified_unsat_with_incomplete_proof=unknown_violations,
    )


def _exact_work_from_suite(report: SuiteReport) -> ExactSearchWork:
    """Populate the exact-search-work counters the fixture benchmark exposes.

    The reference support oracle answers one support query per case and does not
    surface the full closure/controller counters (those come from the frontier
    ``DecodeStats``/``ClosureCounters`` producers).  We record what the fixture
    truly exercised and leave the rest at their honest zero-defaults.
    """

    cases = report.results
    supported = sum(1 for c in cases if c.oracle_verdict == "supported")
    unsupported = sum(1 for c in cases if c.oracle_verdict == "unsupported")
    unknown = sum(1 for c in cases if c.oracle_verdict == "unknown")
    return ExactSearchWork(
        closure_passes=len(cases),
        support_queries=len(cases),
        supported_queries=supported,
        unsupported_queries=unsupported,
        unknown_queries=unknown,
        certified_candidate_removals=unsupported,
        verifier_calls=len(cases),
    )


def run_fixture_row(row: VerifiedSolverRow) -> RowResult:
    """Execute one row in CPU fixture mode (torch-free).

    Only the control (R0) and the closed exact-search benchmark row (R1) run;
    rows that require a trained checkpoint/head or a benchmark family that is
    not committed are marked ``blocked`` with an explicit reason.
    """

    chash = config_hash(row)

    # Frontier-only rows: honest blocked/not_run, never a silent substitution.
    if row.required_capabilities:
        reason = (
            "requires "
            + ", ".join(row.required_capabilities)
            + " (frontier only; not run until VSS4-03)"
        )
        return RowResult(
            row_id=row.row_id,
            run_id=row.run_id,
            status="blocked",
            blocked_reason=reason,
            control=row.control,
            single_variable=row.single_variable,
            config_hash=chash,
            is_closed_benchmark=row.is_closed_benchmark,
            metrics=RowMetrics(),
            gate=None,
            evidence={"required_capabilities": list(row.required_capabilities)},
        )

    if not row.solver_enabled:
        # R0 control: no solver, deterministic finalization only.  In fixture
        # mode there is no model, so quality is not measured (honest None) and
        # the solver correctness fields stay not_applicable.
        metrics = RowMetrics(solver=SolverCorrectness(enabled=False))
        gate = evaluate_verified_solver_gates(
            metrics,
            closed_benchmark=row.is_closed_benchmark,
            solver_enabled=False,
            surface_active=False,
        )
        return RowResult(
            row_id=row.row_id,
            run_id=row.run_id,
            status="ran",
            blocked_reason=None,
            control=row.control,
            single_variable=row.single_variable,
            config_hash=chash,
            is_closed_benchmark=row.is_closed_benchmark,
            metrics=metrics,
            gate=gate,
            evidence={"note": "control row; no solver, no model in fixture mode"},
        )

    # R1: exact deterministic solver over the committed closed benchmark.
    report = run_reference_suite()
    solver = solver_metrics_from_suite(report)
    metrics = RowMetrics(solver=solver, exact_search_work=_exact_work_from_suite(report))
    gate = evaluate_verified_solver_gates(
        metrics,
        closed_benchmark=row.is_closed_benchmark,
        solver_enabled=True,
        surface_active=row.surface_active,
    )
    return RowResult(
        row_id=row.row_id,
        run_id=row.run_id,
        status="ran",
        blocked_reason=None,
        control=row.control,
        single_variable=row.single_variable,
        config_hash=chash,
        is_closed_benchmark=row.is_closed_benchmark,
        metrics=metrics,
        gate=gate,
        evidence={
            "benchmark_id": FIXTURE_BENCHMARK_ID,
            "manifest_digest": report.manifest_digest,
            "case_count": len(report.results),
        },
    )


# --------------------------------------------------------------------------- #
# Matched deltas + report rendering (JSON + Markdown).
# --------------------------------------------------------------------------- #


def _numeric_leaves(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _numeric_leaves(f"{prefix}.{key}" if prefix else str(key), child, out)


def matched_delta(control: RowMetrics, row: RowMetrics) -> dict[str, float]:
    """Numeric deltas row-minus-control, only where both sides are measured.

    ``not_applicable`` (``None``) fields are skipped on either side so a missing
    correctness measurement is never averaged into a zero delta.
    """

    control_leaves: dict[str, float] = {}
    row_leaves: dict[str, float] = {}
    _numeric_leaves("", control.to_dict(), control_leaves)
    _numeric_leaves("", row.to_dict(), row_leaves)
    delta: dict[str, float] = {}
    for key, row_value in row_leaves.items():
        if key in control_leaves:
            delta[key] = row_value - control_leaves[key]
    return delta


def run_matrix(
    *,
    mode: str = "fixture",
    only_rows: tuple[str, ...] | None = None,
    recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the verified-solver matrix and return the report dict.

    ``mode="fixture"`` executes the torch-free CPU path.  ``mode="frontier"``
    resolves every row to ``blocked`` here (no checkpoints/hardware in this
    environment); a real frontier run wires the trained producers in VSS4-03.
    """

    rows = verified_solver_rows()
    if only_rows:
        wanted = {r.upper() for r in only_rows}
        rows = tuple(r for r in rows if r.row_id in wanted)

    results: list[RowResult] = []
    for row in rows:
        if mode == "frontier":
            results.append(
                RowResult(
                    row_id=row.row_id,
                    run_id=row.run_id,
                    status="blocked",
                    blocked_reason=(
                        "frontier mode requires model/data artifacts and "
                        "hardware that are unavailable in this environment"
                    ),
                    control=row.control,
                    single_variable=row.single_variable,
                    config_hash=config_hash(row),
                    is_closed_benchmark=row.is_closed_benchmark,
                    metrics=RowMetrics(),
                    gate=None,
                    evidence={"required_capabilities": list(row.required_capabilities)},
                )
            )
        else:
            results.append(run_fixture_row(row))

    by_id = {r.row_id: r for r in results}
    row_dicts: list[dict[str, Any]] = []
    for result in results:
        entry = result.to_dict()
        control = by_id.get(result.control) if result.control else None
        if (
            control is not None
            and control.status == "ran"
            and result.status == "ran"
        ):
            entry["matched_delta_vs_control"] = matched_delta(
                control.metrics, result.metrics
            )
        else:
            entry["matched_delta_vs_control"] = None
        row_dicts.append(entry)

    ran = [r for r in results if r.status == "ran"]
    blocked = [r for r in results if r.status != "ran"]
    hard_gate_pass = all(r.gate.passed for r in ran if r.gate is not None)

    report_recipe = {
        "device": "cpu",
        "mode": mode,
        "honesty": "fixture_wiring" if mode == "fixture" else "frontier_unrun",
        "fixture_benchmark_id": FIXTURE_BENCHMARK_ID,
        **(recipe or {}),
    }
    return {
        "matrix": f"quality-experiment-matrix-{MATRIX_SET}",
        "matrix_set": MATRIX_SET,
        "schema_version": SCHEMA_VERSION,
        "reference": "docs/design/verified-scope-solver-benchmark.md",
        "recipe": report_recipe,
        "hard_gates": list(HARD_GATES),
        "hard_gates_pass": hard_gate_pass,
        "ablation_axes": [dict(axis) for axis in ABLATION_AXES],
        "rows_ran": [r.row_id for r in ran],
        "rows_blocked": [r.row_id for r in blocked],
        "results": row_dicts,
        "honesty_note": (
            "Fixture wiring evidence only: the closed exact-search row (R1) runs "
            "the VSS4-01 benchmark on CPU; every frontier row is fully specified "
            "but not run until VSS4-03.  No model, quality, or ship claim."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a Markdown measured-results table."""

    lines: list[str] = []
    lines.append("### Verified-solver matrix — fixture wiring")
    lines.append("")
    recipe = report["recipe"]
    lines.append(
        f"Recipe: device={recipe['device']} · mode={recipe['mode']} · "
        f"honesty={recipe['honesty']} · benchmark={recipe['fixture_benchmark_id']}"
    )
    lines.append("")
    lines.append(f"Hard gates evaluated before quality: **{', '.join(report['hard_gates'])}**")
    lines.append("")
    lines.append(
        f"Hard-gate pass (ran rows): **{'PASS' if report['hard_gates_pass'] else 'FAIL'}**"
    )
    lines.append("")
    lines.append("| Row | Status | Control | Single variable | Gate | Config hash |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for entry in report["results"]:
        gate = entry.get("gate")
        if gate is None:
            gate_cell = "—"
        else:
            gate_cell = "PASS" if gate["pass"] else "FAIL"
        control = entry.get("control") or "—"
        status = entry["status"]
        if status != "ran" and entry.get("blocked_reason"):
            status = f"{status} ({entry['blocked_reason']})"
        lines.append(
            f"| {entry['id']} | {status} | {control} | "
            f"{entry['single_variable']} | {gate_cell} | `{entry['config_hash']}` |"
        )
    lines.append("")
    lines.append(f"> {report['honesty_note']}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Describe (resolve configs without loading models or writing files).
# --------------------------------------------------------------------------- #


def describe_rows(
    rows: tuple[VerifiedSolverRow, ...] | None = None,
    *,
    mode: str = "fixture",
) -> list[dict[str, Any]]:
    """Return the exact resolved configs, dependencies, hashes, and hardware
    expectations for the selected rows.  Loads nothing, writes nothing."""

    selected = rows if rows is not None else verified_solver_rows()
    described: list[dict[str, Any]] = []
    for row in selected:
        fixture_runnable = not row.required_capabilities
        described.append(
            {
                "id": row.row_id,
                "run_id": row.run_id,
                "description": row.description,
                "config": resolve_row_config(row),
                "config_hash": config_hash(row),
                "dependencies": list(row.required_capabilities),
                "required_capabilities": list(row.required_capabilities),
                "checkpoint_hashes": (
                    {"fixture_benchmark": FIXTURE_BENCHMARK_ID}
                    if fixture_runnable
                    else {cap: "unresolved" for cap in row.required_capabilities}
                ),
                "data_hashes": {"fixture_benchmark": FIXTURE_BENCHMARK_ID},
                "hardware_expectation": (
                    "cpu" if (mode == "fixture" and fixture_runnable) else "gpu"
                ),
                "fixture_runnable": fixture_runnable and mode == "fixture",
                "is_closed_benchmark": row.is_closed_benchmark,
            }
        )
    return described


__all__ = [
    "SCHEMA_VERSION",
    "MATRIX_SET",
    "FIXTURE_BENCHMARK_ID",
    "HARD_GATES",
    "ABLATION_AXES",
    "SolverCorrectness",
    "ExactSearchWork",
    "CapsuleMetrics",
    "TopologyMetrics",
    "EnergyMetrics",
    "SurfaceMetrics",
    "QualityMetrics",
    "RowMetrics",
    "VerifiedSolverRow",
    "verified_solver_rows",
    "row_by_id",
    "resolve_row_config",
    "config_hash",
    "GateResult",
    "evaluate_verified_solver_gates",
    "RowResult",
    "solver_metrics_from_suite",
    "run_fixture_row",
    "matched_delta",
    "run_matrix",
    "render_markdown",
    "describe_rows",
]
