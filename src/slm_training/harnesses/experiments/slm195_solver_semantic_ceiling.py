"""SLM-195 (FFE3-04): solver-only semantic ceiling harness.

A CPU-only, learned-model-free ceiling experiment over the VSS finite fixture
from ``slm_training.harnesses.solver_bench``.  Every arm is a deterministic
search strategy (DFS with various rankers, BFS, A*, beam search) over the same
expander/verifier oracle.  The only accepted terminal is ``"aa"``.

This harness measures how quickly a perfect solver can close the exact edit
space, providing an upper-bound reference before any learned-model claims.
"""

from __future__ import annotations

import heapq
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from slm_training.dsl.solver.closure import EnumerativeSupportProvider, exact_closure
from slm_training.dsl.solver.controller import (
    CandidateRanker,
    SearchStatus,
    TerminalChecker,
    TerminalOutcome,
    default_hole_selector,
    search,
)
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState
from slm_training.dsl.solver.support import ExpandStatus, ExpandStep, VerifyStatus
from slm_training.harnesses.solver_bench import build_reference_fixture
from slm_training.versioning import build_version_stamp

MANIFEST_SCHEMA = "solver_ceiling_manifest/v1"
EXPERIMENT_ID = "slm195-solver-only-semantic-ceiling"
MATRIX_SET = "slm195_solver_only_semantic_ceiling"
MATRIX_VERSION = "ffe3-04-v1"
ARM_NAMES = (
    "canonical_dfs",
    "random_order",
    "oracle_order",
    "bfs_min_edits",
    "astar_admissible",
    "beam_symbolic",
    "search_work_energy",
)
TERMINATION_STATUSES = (
    "SOLVED_ACCEPTED",
    "CERTIFIED_UNSAT",
    "UNKNOWN_BUDGET",
    "NO_ACCEPTABLE_WITHIN_ENUMERATED_FRONTIER",
    "ABSTAINED_UTILITY",
    "VERIFIER_UNAVAILABLE",
)
DISPOSITIONS = ("solver_ceiling_established", "inconclusive")
UNKNOWN = "UNKNOWN"
REPO_URL = "https://github.com/Tyler-R-Kendrick/slm-training.git"

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, GPU, or ship-gate claim is involved.",
    "The reference fixture is the committed VSS finite word tree; the only verifier-accepted terminal is 'aa'.",
    "Symbolic rankers (including search_work_energy) are deterministic stand-ins for a learned energy model.",
    "A* and beam heuristics are admissible w.r.t. remaining decisions, not a learned value function.",
    "Random-order results depend on the manifest random_seed and are expected to vary.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {k: _to_tuple(v) for k, v in value.items()}
    return value


class _FixtureExpanderAdapter:
    """Make the benchmark fixture expander work with refined/with_decision states.

    The VSS reference expander stores a private ``fingerprint -> prefix`` map for
    states it creates.  Generic controller operations (closure, refinement,
    ``with_decision``) produce states whose hard content matches one of those
    prefixes, but whose fingerprint is not pre-registered.  This adapter derives
    the prefix from the state's ``node`` metadata (or from the parent fingerprint
    plus the chosen value when metadata alone is ambiguous) and registers the
    fingerprint on demand before delegating to the original expander.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.problem_id = inner.problem_id
        self.pack_id = inner.pack_id
        self.constraint_version = inner.constraint_version
        self.bounds = inner.bounds

    @staticmethod
    def _node_prefix(state: FiniteDomainState) -> str | None:
        if not state.holes:
            return None
        for key, value in state.holes[0].metadata:
            if key == "node":
                return "" if value == "ROOT" else str(value)
        return None

    def _infer_prefix(self, state: FiniteDomainState, value: DomainValue) -> str:
        node_prefix = self._node_prefix(state)
        if node_prefix is not None and node_prefix != "":
            return node_prefix
        parent_prefix = ""
        if state.parent_fingerprint is not None:
            parent_prefix = self._inner._prefix_by_fp.get(state.parent_fingerprint, "")
        if (
            state.holes
            and len(state.holes[0].values) == 1
            and state.parent_fingerprint is not None
        ):
            return parent_prefix + str(value.payload.get("letter", ""))
        return parent_prefix

    def successor(self, state: FiniteDomainState, hole_id: Any, value: DomainValue) -> Any:
        fp = state.fingerprint
        if fp not in self._inner._prefix_by_fp:
            self._inner._prefix_by_fp[fp] = self._infer_prefix(state, value)
        try:
            return self._inner.successor(state, hole_id, value)
        except (KeyError, StopIteration):
            return ExpandStep(ExpandStatus.DEAD, detail="no fixture transition")


def _adapt_fixture(raw: Any) -> Any:
    """Return a fixture-like object whose expander is controller-compatible."""
    from types import SimpleNamespace

    return SimpleNamespace(
        expander=_FixtureExpanderAdapter(raw.expander),
        verifier=raw.verifier,
        state=raw.state,
        hole_id=raw.hole_id,
    )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolverCeilingManifestV1:
    """Immutable manifest for one SLM-195 solver-only ceiling run."""

    schema_version: str = MANIFEST_SCHEMA
    run_id: str = ""
    experiment_id: str = EXPERIMENT_ID
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    repo_url: str = REPO_URL
    source_commit: str = UNKNOWN
    dirty_tree_ok: bool = False
    fixture_pack_id: str = "vss4-fixture-word"
    fixture_constraint_version: str = "v1"
    arms: tuple[str, ...] = ()
    budgets: tuple[int, ...] = ()
    max_wall_seconds: int = 180
    random_seed: int = 0
    timestamps: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    disposition: str | None = None
    version_stamp: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported solver ceiling manifest schema {self.schema_version!r}"
            )
        bad_arms = [a for a in self.arms if a not in ARM_NAMES]
        if bad_arms:
            raise ValueError(f"unsupported arms: {bad_arms}; expected subset of {ARM_NAMES}")
        if any(not isinstance(b, int) or b < 1 for b in self.budgets):
            raise ValueError("budgets must be positive ints")
        if self.disposition is not None and self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"unsupported disposition {self.disposition!r}; expected one of {DISPOSITIONS}"
            )
        if self.max_wall_seconds < 1:
            raise ValueError("max_wall_seconds must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolverCeilingManifestV1":
        mapped = dict(data)
        mapped["arms"] = tuple(mapped.get("arms", ()))
        mapped["budgets"] = tuple(int(b) for b in mapped.get("budgets", ()))
        mapped["notes"] = tuple(mapped.get("notes", ()))
        for key in ("timestamps", "version_stamp"):
            mapped[key] = _to_tuple(mapped.get(key, {}))
        known = {f.name for f in cls.__dataclass_fields__.values()}
        mapped = {k: v for k, v in mapped.items() if k in known}
        return cls(**mapped)

    @classmethod
    def load_json(cls, path: Path | str) -> "SolverCeilingManifestV1":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_ready(self) -> list[str]:
        errors: list[str] = []
        if not self.run_id:
            errors.append("run_id is required")
        if not self.arms:
            errors.append("arms must list at least one arm")
        if not self.budgets:
            errors.append("budgets must list at least one budget")
        if self.source_commit == UNKNOWN:
            errors.append("source_commit is required")
        return errors

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "fixture": {
                "pack_id": self.fixture_pack_id,
                "constraint_version": self.fixture_constraint_version,
            },
            "arms": list(self.arms),
            "budgets": list(self.budgets),
            "random_seed": self.random_seed,
            "max_wall_seconds": self.max_wall_seconds,
            "source_commit": self.source_commit,
            "ready_blockers": self.check_ready(),
            "version_stamp": self.version_stamp,
        }


def build_default_manifest(
    run_id: str,
    *,
    arms: tuple[str, ...] | None = None,
    budgets: tuple[int, ...] | None = None,
    random_seed: int = 0,
    max_wall_seconds: int = 180,
    source_commit: str | None = None,
    dirty_tree_ok: bool | None = None,
    fixture_pack_id: str = "vss4-fixture-word",
    fixture_constraint_version: str = "v1",
) -> SolverCeilingManifestV1:
    """Build a default SLM-195 manifest with current-repo provenance prefilled."""
    stamp = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm195_solver_only_semantic_ceiling",
    )
    commit = source_commit or stamp.get("code_commit") or UNKNOWN
    dirty = dirty_tree_ok if dirty_tree_ok is not None else bool(stamp.get("code_dirty"))
    return SolverCeilingManifestV1(
        run_id=run_id,
        source_commit=commit,
        dirty_tree_ok=dirty,
        fixture_pack_id=fixture_pack_id,
        fixture_constraint_version=fixture_constraint_version,
        arms=arms if arms is not None else ARM_NAMES,
        budgets=budgets if budgets is not None else (10, 100, 1000),
        random_seed=random_seed,
        max_wall_seconds=max_wall_seconds,
        timestamps={"created_at": _now()},
        version_stamp=stamp,
    )


# --------------------------------------------------------------------------- #
# Rankers
# --------------------------------------------------------------------------- #


class CanonicalOrderRanker:
    """Identity ordering over the canonical live domain values."""

    @property
    def ranker_id(self) -> str:
        return "canonical-order-v1"

    def rank(
        self, state: FiniteDomainState, hole_id: Any, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        return tuple(values)


@dataclass(frozen=True)
class RandomRanker:
    """Deterministic shuffle from a fixed seed."""

    seed: int

    @property
    def ranker_id(self) -> str:
        return f"random-{self.seed}"

    def rank(
        self, state: FiniteDomainState, hole_id: Any, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        ordered = list(values)
        random.Random(self.seed).shuffle(ordered)
        return tuple(ordered)


@dataclass(frozen=True)
class OracleRanker:
    """Prefer values whose payload letter appears earlier in ``target_path``."""

    target_path: tuple[str, ...]

    @property
    def ranker_id(self) -> str:
        return "oracle-order-v1"

    def rank(
        self, state: FiniteDomainState, hole_id: Any, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        order = {letter: idx for idx, letter in enumerate(self.target_path)}

        def key(value: DomainValue) -> tuple[int, int]:
            letter = str(value.payload.get("letter", ""))
            return (order.get(letter, len(self.target_path)), values.index(value))

        return tuple(sorted(values, key=key))


@dataclass(frozen=True)
class SearchWorkEnergyRanker:
    """Symbolic stand-in for a learned energy model: prefer lower remaining work."""

    expander: Any | None = None

    @property
    def ranker_id(self) -> str:
        return "search-work-energy-v1"

    def _work(
        self, state: FiniteDomainState, hole_id: Any, value: DomainValue
    ) -> int:
        if self.expander is None:
            return 0
        step = self.expander.successor(state, hole_id, value)
        if step.status is ExpandStatus.TERMINAL:
            return 0
        if step.status is ExpandStatus.DEAD:
            return 1_000_000_000
        if step.status is ExpandStatus.INCOMPLETE:
            return 1_000_000
        child = step.next_state
        if child is None:
            return 1_000_000
        return sum(max(0, len(h.values) - 1) for h in child.holes)

    def rank(
        self, state: FiniteDomainState, hole_id: Any, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        return tuple(
            sorted(values, key=lambda v: (self._work(state, hole_id, v), values.index(v)))
        )


# --------------------------------------------------------------------------- #
# Terminal checker
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FixtureTerminalChecker:
    """Materialize a structurally-solved state and run the fixture verifier."""

    expander: Any
    verifier: Any

    def check(self, state: FiniteDomainState) -> TerminalOutcome:
        if not state.is_structurally_solved:
            return TerminalOutcome(
                accepted=False,
                source=None,
                report=None,
                detail="state is not structurally solved",
            )
        hole = state.holes[0]
        value = hole.values[0]
        step = self.expander.successor(state, hole.hole_id, value)
        if step.status is not ExpandStatus.TERMINAL:
            return TerminalOutcome(
                accepted=False,
                source=None,
                report={"expand_status": step.status.value},
                detail=f"expected terminal expand status, got {step.status.value}",
            )
        program = step.program or ""
        outcome = self.verifier.verify(program)
        accepted = outcome.status is VerifyStatus.ACCEPT
        return TerminalOutcome(
            accepted=accepted,
            source=program,
            report={
                "program": program,
                "verifier_status": outcome.status.value,
                "detail": outcome.detail,
            },
            detail="accepted" if accepted else f"rejected: {outcome.detail}",
        )


# --------------------------------------------------------------------------- #
# Arm results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmResult:
    """Outcome of one arm × budget cell."""

    arm_name: str
    budget: int
    status: str
    terminal_program: str | None
    decisions: int
    wall_seconds: float
    counters: dict[str, int]
    stop_reason: str | None
    error: str = ""

    def __post_init__(self) -> None:
        if self.status not in TERMINATION_STATUSES:
            raise ValueError(f"unsupported status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "budget": self.budget,
            "status": self.status,
            "terminal_program": self.terminal_program,
            "decisions": self.decisions,
            "wall_seconds": self.wall_seconds,
            "counters": dict(self.counters),
            "stop_reason": self.stop_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArmResult":
        return cls(
            arm_name=str(data["arm_name"]),
            budget=int(data["budget"]),
            status=str(data["status"]),
            terminal_program=data.get("terminal_program"),
            decisions=int(data["decisions"]),
            wall_seconds=float(data["wall_seconds"]),
            counters=dict(data.get("counters", {})),
            stop_reason=data.get("stop_reason"),
            error=str(data.get("error", "")),
        )


def _map_search_status(
    search_status: SearchStatus, source: str | None, stop_reason: str | None
) -> str:
    if search_status is SearchStatus.SOLVED:
        return "SOLVED_ACCEPTED"
    if search_status is SearchStatus.CERTIFIED_UNSAT:
        return "CERTIFIED_UNSAT"
    if search_status is SearchStatus.BUDGET_EXHAUSTED:
        return "UNKNOWN_BUDGET"
    if source is not None:
        return "NO_ACCEPTABLE_WITHIN_ENUMERATED_FRONTIER"
    return "UNKNOWN_BUDGET"


def _search_counters(result: Any) -> dict[str, int]:
    return dict(result.counters.to_dict()) if hasattr(result, "counters") else {}


def run_dfs_arm(
    arm_name: str,
    state: FiniteDomainState,
    _hole_id: Any,
    provider: Any,
    terminal_checker: TerminalChecker,
    ranker: CandidateRanker,
    max_decisions: int,
) -> ArmResult:
    """Run the bounded closure+branching controller with the supplied ranker."""
    start = time.perf_counter()
    try:
        result = search(
            state,
            provider,
            terminal_checker,
            ranker=ranker,
            max_decisions=max_decisions,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ArmResult(
            arm_name=arm_name,
            budget=max_decisions,
            status="VERIFIER_UNAVAILABLE",
            terminal_program=None,
            decisions=0,
            wall_seconds=time.perf_counter() - start,
            counters={},
            stop_reason=None,
            error=str(exc),
        )
    elapsed = time.perf_counter() - start
    status = _map_search_status(result.status, result.source, result.stop_reason)
    return ArmResult(
        arm_name=arm_name,
        budget=max_decisions,
        status=status,
        terminal_program=result.source,
        decisions=len(result.decisions),
        wall_seconds=elapsed,
        counters=_search_counters(result),
        stop_reason=result.stop_reason,
    )


# --------------------------------------------------------------------------- #
# Breadth-first, A*, and beam arms
# --------------------------------------------------------------------------- #


def _heuristic(state: FiniteDomainState) -> int:
    """Admissible: sum of (domain size - 1) over unresolved holes."""
    return sum(max(0, len(h.values) - 1) for h in state.holes)


def run_bfs_arm(
    state: FiniteDomainState,
    provider: Any,
    terminal_checker: TerminalChecker,
    max_decisions: int,
) -> ArmResult:
    """Breadth-first search over one-step successors with optional closure pruning."""
    start = time.perf_counter()
    root = exact_closure(state, provider).state
    queue: deque[FiniteDomainState] = deque([root])
    visited: set[str] = set()
    counter = 0
    while queue and counter < max_decisions:
        current = queue.popleft()
        fp = current.fingerprint
        if fp in visited:
            continue
        visited.add(fp)
        if current.is_structurally_solved:
            outcome = terminal_checker.check(current)
            counter += 1
            if outcome.accepted:
                return ArmResult(
                    arm_name="bfs_min_edits",
                    budget=max_decisions,
                    status="SOLVED_ACCEPTED",
                    terminal_program=outcome.source,
                    decisions=counter,
                    wall_seconds=time.perf_counter() - start,
                    counters={},
                    stop_reason=None,
                )
            continue
        hole_id = default_hole_selector(current)
        if hole_id is None:
            continue
        for value in current.domain(hole_id).values:
            child = current.with_decision(hole_id, value)
            child = exact_closure(child, provider).state
            counter += 1
            if child.is_bottom:
                continue
            queue.append(child)
            if counter >= max_decisions:
                break
    elapsed = time.perf_counter() - start
    return ArmResult(
        arm_name="bfs_min_edits",
        budget=max_decisions,
        status="UNKNOWN_BUDGET" if counter >= max_decisions else "NO_ACCEPTABLE_WITHIN_ENUMERATED_FRONTIER",
        terminal_program=None,
        decisions=counter,
        wall_seconds=elapsed,
        counters={},
        stop_reason="budget:max_decisions" if counter >= max_decisions else "frontier_exhausted",
    )


def run_astar_arm(
    state: FiniteDomainState,
    provider: Any,
    terminal_checker: TerminalChecker,
    max_decisions: int,
) -> ArmResult:
    """A* search with an admissible heuristic over unresolved candidates."""
    start = time.perf_counter()
    root = exact_closure(state, provider).state
    tie = 0
    heap: list[tuple[int, int, int, FiniteDomainState]] = [
        (_heuristic(root), 0, tie, root)
    ]
    best_g: dict[str, int] = {root.fingerprint: 0}
    counter = 0
    while heap and counter < max_decisions:
        _f, g, _tie, current = heapq.heappop(heap)
        if best_g.get(current.fingerprint, float("inf")) < g:
            continue
        if current.is_structurally_solved:
            outcome = terminal_checker.check(current)
            counter += 1
            if outcome.accepted:
                return ArmResult(
                    arm_name="astar_admissible",
                    budget=max_decisions,
                    status="SOLVED_ACCEPTED",
                    terminal_program=outcome.source,
                    decisions=counter,
                    wall_seconds=time.perf_counter() - start,
                    counters={},
                    stop_reason=None,
                )
            continue
        hole_id = default_hole_selector(current)
        if hole_id is None:
            continue
        for value in current.domain(hole_id).values:
            child = current.with_decision(hole_id, value)
            child = exact_closure(child, provider).state
            if child.is_bottom:
                continue
            ng = g + 1
            if ng < best_g.get(child.fingerprint, float("inf")):
                best_g[child.fingerprint] = ng
                tie += 1
                heapq.heappush(heap, (ng + _heuristic(child), ng, tie, child))
                counter += 1
                if counter >= max_decisions:
                    break
    elapsed = time.perf_counter() - start
    return ArmResult(
        arm_name="astar_admissible",
        budget=max_decisions,
        status="UNKNOWN_BUDGET" if counter >= max_decisions else "NO_ACCEPTABLE_WITHIN_ENUMERATED_FRONTIER",
        terminal_program=None,
        decisions=counter,
        wall_seconds=elapsed,
        counters={},
        stop_reason="budget:max_decisions" if counter >= max_decisions else "frontier_exhausted",
    )


def run_beam_arm(
    state: FiniteDomainState,
    provider: Any,
    terminal_checker: TerminalChecker,
    max_decisions: int,
    beam_width: int = 2,
) -> ArmResult:
    """Beam search ranked by the admissible unresolved-candidate heuristic."""
    start = time.perf_counter()
    root = exact_closure(state, provider).state
    layer = [root]
    counter = 0
    while layer and counter < max_decisions:
        successors: list[FiniteDomainState] = []
        for current in layer:
            if current.is_structurally_solved:
                outcome = terminal_checker.check(current)
                counter += 1
                if outcome.accepted:
                    return ArmResult(
                        arm_name="beam_symbolic",
                        budget=max_decisions,
                        status="SOLVED_ACCEPTED",
                        terminal_program=outcome.source,
                        decisions=counter,
                        wall_seconds=time.perf_counter() - start,
                        counters={},
                        stop_reason=None,
                    )
                continue
            hole_id = default_hole_selector(current)
            if hole_id is None:
                continue
            for value in current.domain(hole_id).values:
                child = current.with_decision(hole_id, value)
                child = exact_closure(child, provider).state
                if child.is_bottom:
                    continue
                successors.append(child)
                counter += 1
                if counter >= max_decisions:
                    break
            if counter >= max_decisions:
                break
        if not successors:
            break
        scored = sorted(successors, key=lambda s: (_heuristic(s), s.fingerprint))
        seen: set[str] = set()
        layer = []
        for s in scored:
            if s.fingerprint not in seen and len(layer) < beam_width:
                seen.add(s.fingerprint)
                layer.append(s)
    elapsed = time.perf_counter() - start
    return ArmResult(
        arm_name="beam_symbolic",
        budget=max_decisions,
        status="UNKNOWN_BUDGET" if counter >= max_decisions else "NO_ACCEPTABLE_WITHIN_ENUMERATED_FRONTIER",
        terminal_program=None,
        decisions=counter,
        wall_seconds=elapsed,
        counters={},
        stop_reason="budget:max_decisions" if counter >= max_decisions else "frontier_exhausted",
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolverCeilingReport:
    """Aggregated result of an SLM-195 ceiling matrix."""

    schema_version: str = "solver_ceiling_report/v1"
    run_id: str = ""
    experiment_id: str = EXPERIMENT_ID
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    fixture_pack_id: str = "vss4-fixture-word"
    fixture_constraint_version: str = "v1"
    arms: tuple[str, ...] = ()
    budgets: tuple[int, ...] = ()
    random_seed: int = 0
    source_commit: str = UNKNOWN
    disposition: str = "inconclusive"
    timestamps: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    rows: tuple[ArmResult, ...] = ()
    version_stamp: Mapping[str, Any] = field(default_factory=dict)
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "fixture_pack_id": self.fixture_pack_id,
            "fixture_constraint_version": self.fixture_constraint_version,
            "arms": list(self.arms),
            "budgets": list(self.budgets),
            "random_seed": self.random_seed,
            "source_commit": self.source_commit,
            "disposition": self.disposition,
            "timestamps": dict(self.timestamps),
            "notes": list(self.notes),
            "rows": [r.to_dict() for r in self.rows],
            "version_stamp": dict(self.version_stamp),
            "honest_caveats": list(self.honest_caveats),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False, default=str
        )

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolverCeilingReport":
        return cls(
            schema_version=str(data.get("schema_version", "solver_ceiling_report/v1")),
            run_id=str(data.get("run_id", "")),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            fixture_pack_id=str(data.get("fixture_pack_id", "vss4-fixture-word")),
            fixture_constraint_version=str(data.get("fixture_constraint_version", "v1")),
            arms=tuple(data.get("arms", ())),
            budgets=tuple(int(b) for b in data.get("budgets", ())),
            random_seed=int(data.get("random_seed", 0)),
            source_commit=str(data.get("source_commit", UNKNOWN)),
            disposition=str(data.get("disposition", "inconclusive")),
            timestamps=dict(data.get("timestamps", {})),
            notes=tuple(data.get("notes", ())),
            rows=tuple(ArmResult.from_dict(r) for r in data.get("rows", [])),
            version_stamp=dict(data.get("version_stamp", {})),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
        )

    @classmethod
    def load_json(cls, path: Path | str) -> "SolverCeilingReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def render_markdown(report: SolverCeilingReport) -> str:
    """Render a human-readable measured-results summary."""
    lines: list[str] = [
        "# SLM-195 (FFE3-04) solver-only semantic ceiling",
        "",
        f"- **Run ID:** {report.run_id}",
        f"- **Experiment:** {report.experiment_id}",
        f"- **Matrix set:** {report.matrix_set}",
        f"- **Matrix version:** {report.matrix_version}",
        f"- **Fixture:** {report.fixture_pack_id}/{report.fixture_constraint_version}",
        f"- **Source commit:** {report.source_commit}",
        f"- **Random seed:** {report.random_seed}",
        f"- **Disposition:** {report.disposition}",
        f"- **Timestamp:** {report.timestamps.get('finished_at', 'unknown')}",
        "",
        "## Arms",
        "",
        "| arm | budget | status | decisions | terminal | wall (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        terminal = f"`{row.terminal_program}`" if row.terminal_program else "—"
        lines.append(
            f"| {row.arm_name} | {row.budget} | {row.status} | {row.decisions} | "
            f"{terminal} | {row.wall_seconds:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Budget grid",
            "",
        ]
    )
    for budget in report.budgets:
        lines.append(f"### budget={budget}")
        lines.append("")
        for row in report.rows:
            if row.budget == budget:
                lines.append(f"- **{row.arm_name}**: {row.status} ({row.decisions} decisions)")
        lines.append("")
    lines.extend(
        [
            "## Honest caveats",
            "",
        ]
    )
    for caveat in report.honest_caveats:
        lines.append(f"- {caveat}")
    lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "This harness establishes a solver-only ceiling on the exact VSS finite fixture. "
            "Any learned model that claims to improve on these symbolic baselines must be "
            "evaluated against the same fixture and proven not to introduce false UNSAT or "
            "unsupported candidate deletions.",
            "",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _ranker_for_arm(arm: str, fixture: Any, random_seed: int) -> CandidateRanker:
    if arm in {"canonical_dfs", "bfs_min_edits", "astar_admissible", "beam_symbolic"}:
        return CanonicalOrderRanker()
    if arm == "random_order":
        return RandomRanker(random_seed)
    if arm == "oracle_order":
        return OracleRanker(target_path=("a", "a"))
    if arm == "search_work_energy":
        return SearchWorkEnergyRanker(expander=fixture.expander)
    raise ValueError(f"no ranker configured for arm {arm!r}")


def _run_arm(
    arm: str,
    budget: int,
    fixture: Any,
    provider: Any,
    terminal_checker: TerminalChecker,
    random_seed: int,
) -> ArmResult:
    if arm in {"canonical_dfs", "random_order", "oracle_order", "search_work_energy"}:
        ranker = _ranker_for_arm(arm, fixture, random_seed)
        return run_dfs_arm(
            arm,
            fixture.state,
            fixture.hole_id,
            provider,
            terminal_checker,
            ranker,
            budget,
        )
    if arm == "bfs_min_edits":
        return run_bfs_arm(fixture.state, provider, terminal_checker, budget)
    if arm == "astar_admissible":
        return run_astar_arm(fixture.state, provider, terminal_checker, budget)
    if arm == "beam_symbolic":
        return run_beam_arm(fixture.state, provider, terminal_checker, budget)
    raise ValueError(f"unknown arm {arm!r}")


def _write_design_docs(report: SolverCeilingReport) -> None:
    design_dir = Path("docs/design")
    design_dir.mkdir(parents=True, exist_ok=True)
    date = _today_yyyymmdd()
    json_path = design_dir / f"iter-slm195-solver-only-semantic-ceiling-{date}.json"
    md_path = design_dir / f"iter-slm195-solver-only-semantic-ceiling-{date}.md"
    report.write_json(json_path)
    md_path.write_text(render_markdown(report), encoding="utf-8")


def run_ceiling(manifest: SolverCeilingManifestV1) -> SolverCeilingReport:
    """Run every configured arm × budget and persist design artifacts."""
    fixture = _adapt_fixture(build_reference_fixture())
    provider = EnumerativeSupportProvider(fixture.expander, fixture.verifier)
    terminal_checker = FixtureTerminalChecker(fixture.expander, fixture.verifier)

    start = time.perf_counter()
    rows: list[ArmResult] = []
    for arm in manifest.arms:
        for budget in manifest.budgets:
            rows.append(
                _run_arm(arm, budget, fixture, provider, terminal_checker, manifest.random_seed)
            )
    elapsed = time.perf_counter() - start

    solved = any(r.status == "SOLVED_ACCEPTED" for r in rows)
    disposition = "solver_ceiling_established" if solved else "inconclusive"
    report = SolverCeilingReport(
        run_id=manifest.run_id,
        arms=manifest.arms,
        budgets=manifest.budgets,
        random_seed=manifest.random_seed,
        source_commit=manifest.source_commit,
        disposition=disposition,
        fixture_pack_id=manifest.fixture_pack_id,
        fixture_constraint_version=manifest.fixture_constraint_version,
        timestamps={"started_at": _now(), "finished_at": _now()},
        notes=(
            f"ran {len(manifest.arms)} arms × {len(manifest.budgets)} budgets "
            f"in {elapsed:.6f}s",
        ),
        rows=tuple(rows),
        version_stamp=manifest.version_stamp,
    )
    _write_design_docs(report)
    return report
