"""Regression tests for the VSS2-02 capsule solver coordinator."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from slm_training.data.progspec.capsules import (
    CapsuleGraph,
    DependencyKind,
    ScopeEdge,
    ScopeNode,
    VerificationCapsule,
)
from slm_training.dsl.pack import DslPack, PackSlotUnavailable, get_backend
from slm_training.dsl.solver.capsule_solver import (
    BindingSummary,
    CapsuleInterfaceSummary,
    CapsuleProblem,
    CapsuleSolveResult,
    ExternalInput,
    SlotSummary,
    build_capsule_solve_plan,
    solve_capsule_graph,
)
from slm_training.dsl.solver.closure import EnumerativeSupportProvider
from slm_training.dsl.solver.controller import TerminalChecker, TerminalOutcome
from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleDomain, HoleId, SolverBounds
from slm_training.dsl.solver.support import (
    ExpandStatus,
    ExpandStep,
    Verifier,
    VerifyOutcome,
    VerifyStatus,
)


@dataclass
class CapsuleSpec:
    """Test-only description of one fake finite capsule."""

    holes: dict[str, list[int]]
    constraints: Callable[[dict[str, int]], bool] | None = None
    outputs: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Fake finite pack helpers
# --------------------------------------------------------------------------- #


def _hole_id(capsule_id: str, name: str) -> HoleId:
    return HoleId(namespace=capsule_id, path=(name,), kind="var")


def _int_value(value: int) -> DomainValue:
    return DomainValue.create("int", value)


class FakeVerifier(Verifier):
    """Accepts every terminal program; constraints are enforced by the expander."""

    profile = "fake-verifier-v1"

    def verify(self, program: str) -> VerifyOutcome:
        return VerifyOutcome(status=VerifyStatus.ACCEPT, detail="ok")


class FakeExpander:
    """Deterministic expander for one fake capsule."""

    def __init__(self, capsule_id: str, spec: CapsuleSpec, bounds: SolverBounds) -> None:
        self.problem_id = capsule_id
        self.pack_id = "fake"
        self.constraint_version = "v1"
        self.bounds = bounds
        self.spec = spec

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        after = state.with_decision(hole_id, value)
        if after.is_structurally_solved:
            assignment: dict[str, int] = {}
            for hole in after.holes:
                name = hole.hole_id.path[0]
                assignment[name] = hole.values[0].payload
            ok = self.spec.constraints is None or self.spec.constraints(assignment)
            if not ok:
                return ExpandStep(
                    status=ExpandStatus.DEAD, detail="constraint_violation"
                )
            program = f"{self.problem_id}:{assignment}"
            return ExpandStep(
                status=ExpandStatus.TERMINAL, program=program, detail="ok"
            )
        return ExpandStep(
            status=ExpandStatus.CONTINUE, next_state=after, coverage="complete"
        )


class FakeBuilder:
    """Builds a CapsuleProblem from the SPECS registry."""

    def __init__(self, specs: dict[str, CapsuleSpec]) -> None:
        self.specs = specs

    def build_problem(
        self,
        capsule: VerificationCapsule,
        predecessor_summaries: tuple[CapsuleInterfaceSummary, ...],
        external_inputs: tuple[ExternalInput, ...],
        bounds: SolverBounds,
    ) -> CapsuleProblem:
        spec = self.specs[capsule.capsule_id]
        allowed: dict[str, list[int]] = {
            name: list(values) for name, values in spec.holes.items()
        }

        # Predecessor outputs become singleton inputs.
        for summary in predecessor_summaries:
            for binding in summary.output_bindings:
                if binding.value is not None and binding.name in allowed:
                    allowed[binding.name] = [binding.value.payload]

        # External slots are listed but carry no value in these fixtures.
        for ext in external_inputs:
            if ext.name in allowed:
                # keep the declared domain; the slot name is merely recorded
                pass

        holes = tuple(
            HoleDomain(
                hole_id=_hole_id(capsule.capsule_id, name),
                values=tuple(_int_value(v) for v in sorted(allowed[name])),
            )
            for name in sorted(allowed)
        )
        state = FiniteDomainState(
            problem_id=capsule.capsule_id,
            pack_id="fake",
            constraint_version="v1",
            bounds=bounds,
            holes=holes,
        )
        return CapsuleProblem(
            capsule=capsule,
            state=state,
            predecessor_summaries=predecessor_summaries,
            external_inputs=external_inputs,
        )


class FakeSummaryExtractor:
    """Extracts output bindings from solved singleton holes."""

    def __init__(self, specs: dict[str, CapsuleSpec]) -> None:
        self.specs = specs

    def extract_summary(
        self, capsule: VerificationCapsule, state: FiniteDomainState
    ) -> CapsuleInterfaceSummary:
        spec = self.specs[capsule.capsule_id]
        input_bindings = tuple(
            BindingSummary(name=name, kind="input") for name in spec.inputs
        )
        output_bindings = []
        for name in spec.outputs:
            hole_id = _hole_id(capsule.capsule_id, name)
            domain = state.domain(hole_id)
            value = domain.values[0] if domain.values else None
            output_bindings.append(
                BindingSummary(name=name, kind="output", value=value)
            )
        slots = tuple(
            SlotSummary(name=name) for name in sorted(capsule.external_dependencies)
        )
        summary = CapsuleInterfaceSummary(
            capsule_id=capsule.capsule_id,
            input_bindings=input_bindings,
            output_bindings=tuple(output_bindings),
            slots=slots,
            preconditions=(),
            postconditions=(),
            effects=(),
            exceptions=(),
            captures=(),
            conservative=False,
            fingerprint="0" * 64,  # replaced below
        )
        payload = summary.to_dict()
        payload.pop("fingerprint")
        import hashlib
        import json

        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CapsuleInterfaceSummary(
            capsule_id=summary.capsule_id,
            input_bindings=summary.input_bindings,
            output_bindings=summary.output_bindings,
            slots=summary.slots,
            preconditions=summary.preconditions,
            postconditions=summary.postconditions,
            effects=summary.effects,
            exceptions=summary.exceptions,
            captures=summary.captures,
            conservative=summary.conservative,
            fingerprint=fingerprint,
        )


class FakeTerminalChecker(TerminalChecker):
    """Local terminal checker accepts any structurally solved state."""

    def check(self, state: FiniteDomainState) -> TerminalOutcome:
        return TerminalOutcome(accepted=True, source="local", report=None)


def make_materializer() -> Callable[[tuple[Any, ...]], str]:
    def materialize(results: tuple[Any, ...]) -> str:
        parts = []
        for result in sorted(results, key=lambda r: r.capsule_id):
            if result.status != "solved":
                continue
            summary = result.summary
            assert summary is not None
            out = {
                b.name: b.value.payload if b.value is not None else None
                for b in summary.output_bindings
            }
            parts.append(f"{result.capsule_id}={out}")
        return ";".join(parts)

    return materialize


def make_global_verifier(reject_if: Callable[[str | None], bool]) -> Callable[[str | None], TerminalOutcome]:
    def verify(source: str | None) -> TerminalOutcome:
        if source is None or reject_if(source):
            return TerminalOutcome(
                accepted=False, detail="global_rejected", report={"source": source}
            )
        return TerminalOutcome(
            accepted=True, detail="global_accepted", report={"source": source}
        )

    return verify


class MultiSupportProvider:
    """Dispatches to the per-capsule enumerative provider by problem_id."""

    def __init__(self, specs: dict[str, CapsuleSpec], bounds: SolverBounds) -> None:
        self.specs = specs
        self.bounds = bounds
        self._providers: dict[str, EnumerativeSupportProvider] = {}

    @property
    def backend_version(self) -> str:
        return "fake-multi-v1"

    def _provider(self, problem_id: str) -> EnumerativeSupportProvider:
        if problem_id not in self._providers:
            self._providers[problem_id] = EnumerativeSupportProvider(
                FakeExpander(problem_id, self.specs[problem_id], self.bounds),
                FakeVerifier(),
            )
        return self._providers[problem_id]

    def check(self, state: FiniteDomainState, query: Any) -> Any:
        return self._provider(state.problem_id).check(state, query)

    def replay(self, certificate: Any, *, state: FiniteDomainState) -> Any:
        return self._provider(state.problem_id).replay(certificate, state=state)


def _provider(specs: dict[str, CapsuleSpec], bounds: SolverBounds) -> MultiSupportProvider:
    return MultiSupportProvider(specs, bounds)


# --------------------------------------------------------------------------- #
# Graph builders
# --------------------------------------------------------------------------- #


def _root_node(spec_id: str) -> ScopeNode:
    return ScopeNode(
        node_id=f"{spec_id}:root",
        scope_id=None,
        kind="root",
        ast_path=(),
        member_paths=(),
        definitions=(),
        external_dependencies=(),
    )


def _statement_node(spec_id: str, name: str) -> ScopeNode:
    return ScopeNode(
        node_id=f"{spec_id}:{name}",
        scope_id=f"{spec_id}:{name}",
        kind="statement",
        ast_path=(name,),
        member_paths=(),
        definitions=(name,) if not name.startswith(":") else (),
        external_dependencies=(),
    )


def _capsule(capsule_id: str, node_ids: tuple[str, ...]) -> VerificationCapsule:
    return VerificationCapsule(
        capsule_id=capsule_id,
        node_ids=node_ids,
        entry_node_id=node_ids[0],
        external_dependencies=(),
    )


def _make_graph(
    root: ScopeNode,
    nodes: tuple[ScopeNode, ...],
    edges: tuple[ScopeEdge, ...],
    capsules: tuple[VerificationCapsule, ...],
    spec_id: str,
) -> CapsuleGraph:
    return CapsuleGraph(
        root_id=root.node_id,
        nodes=nodes,
        edges=edges,
        capsules=capsules,
        spec_id=spec_id,
        version=CapsuleGraph.VERSION,
    )


def _ref_edge(source: str, target: str) -> ScopeEdge:
    return ScopeEdge(
        source=source, target=target, kind=DependencyKind.REFERENCE, role="ref"
    )


def _external_edge(source: str, role: str) -> ScopeEdge:
    return ScopeEdge(
        source=source, target=f"{source.split(':')[0]}:root", kind=DependencyKind.EXTERNAL, role=role
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_build_plan_stages_are_topological():
    spec_id = "plan-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    b = _statement_node(spec_id, "b")
    c = _statement_node(spec_id, "c")
    nodes = (root, a, b, c)
    edges = (
        _ref_edge(a.node_id, b.node_id),
        _ref_edge(b.node_id, c.node_id),
    )
    capsules = (
        _capsule(f"{spec_id}:ca", (a.node_id,)),
        _capsule(f"{spec_id}:cb", (b.node_id,)),
        _capsule(f"{spec_id}:cc", (c.node_id,)),
    )
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=edges,
            capsules=capsules,
            spec_id=spec_id,
        )
    plan = build_capsule_solve_plan(graph)
    assert plan.stages == (
        (f"{spec_id}:ca",),
        (f"{spec_id}:cb",),
        (f"{spec_id}:cc",),
    )
    assert plan.fingerprint != plan.graph_fingerprint


def test_build_plan_groups_independent_capsules():
    spec_id = "independent-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    b = _statement_node(spec_id, "b")
    nodes = (root, a, b)
    edges = ()
    capsules = (
        _capsule(f"{spec_id}:ca", (a.node_id,)),
        _capsule(f"{spec_id}:cb", (b.node_id,)),
    )
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=edges,
            capsules=capsules,
            spec_id=spec_id,
        )
    plan = build_capsule_solve_plan(graph)
    assert len(plan.stages) == 1
    assert set(plan.stages[0]) == {f"{spec_id}:ca", f"{spec_id}:cb"}


def test_two_node_scc_is_joint_problem():
    spec_id = "scc-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    b = _statement_node(spec_id, "b")
    nodes = (root, a, b)
    # mutual reference creates a 2-node SCC
    edges = (
        _ref_edge(a.node_id, b.node_id),
        _ref_edge(b.node_id, a.node_id),
    )
    capsules = (_capsule(f"{spec_id}:c_ab", (a.node_id, b.node_id)),)
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=edges,
            capsules=capsules,
            spec_id=spec_id,
        )
    plan = build_capsule_solve_plan(graph)
    assert plan.joint_sccs == ((a.node_id, b.node_id),)

    specs = {
        capsules[0].capsule_id: CapsuleSpec(
            holes={"a": [0, 1], "b": [0, 1]},
            constraints=lambda assn: assn["a"] != assn["b"],
            outputs=("a", "b"),
        )
    }
    bounds = SolverBounds(
        max_tokens=1000,
        max_nodes=1000,
        max_depth=10,
        max_backtracks=100,
        max_verifier_calls=100,
    )
    result = solve_capsule_graph(
        graph,
        builder=FakeBuilder(specs),
        provider=_provider(specs, bounds),
        terminal_checker=FakeTerminalChecker(),
        summary_extractor=FakeSummaryExtractor(specs),
        materializer=make_materializer(),
        global_verifier=make_global_verifier(lambda _: False),
        bounds=bounds,
    )
    assert result.status == "solved"
    assert result.counters.joint_count == 1
    assert len(result.capsule_results[0].search_result.decisions) >= 1
    # Cyclic dependencies were submitted as one joint problem, not two separate ones.
    assert len(result.capsule_results) == 1


def test_predecessor_output_becomes_successor_input():
    spec_id = "chain-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    b = _statement_node(spec_id, "b")
    nodes = (root, a, b)
    edges = (_ref_edge(a.node_id, b.node_id),)
    ca = _capsule(f"{spec_id}:ca", (a.node_id,))
    cb = _capsule(f"{spec_id}:cb", (b.node_id,))
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=edges,
            capsules=(ca, cb),
            spec_id=spec_id,
        )
    specs = {
        ca.capsule_id: CapsuleSpec(
            holes={"a": [1, 2, 3]},
            constraints=lambda assn: assn["a"] >= 2,
            outputs=("a",),
        ),
        cb.capsule_id: CapsuleSpec(
            holes={"a": [1, 2, 3], "b": [1, 2, 3]},
            constraints=lambda assn: assn["b"] == assn["a"],
            outputs=("b",),
        ),
    }
    bounds = SolverBounds(
        max_tokens=1000,
        max_nodes=1000,
        max_depth=10,
        max_backtracks=100,
        max_verifier_calls=100,
    )
    result = solve_capsule_graph(
        graph,
        builder=FakeBuilder(specs),
        provider=_provider(specs, bounds),
        terminal_checker=FakeTerminalChecker(),
        summary_extractor=FakeSummaryExtractor(specs),
        materializer=make_materializer(),
        global_verifier=make_global_verifier(lambda _: False),
        bounds=bounds,
    )
    assert result.status == "solved"
    ca_result = result.capsule_results[0]
    cb_result = result.capsule_results[1]
    a_value = next(
        b.value.payload for b in ca_result.summary.output_bindings if b.name == "a"
    )
    assert a_value == 2
    b_value = next(
        b.value.payload for b in cb_result.summary.output_bindings if b.name == "b"
    )
    assert b_value == a_value


def test_unknown_predecessor_summary_blocks_successor():
    spec_id = "conservative-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    b = _statement_node(spec_id, "b")
    nodes = (root, a, b)
    edges = (_ref_edge(a.node_id, b.node_id),)
    ca = _capsule(f"{spec_id}:ca", (a.node_id,))
    cb = _capsule(f"{spec_id}:cb", (b.node_id,))
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=edges,
            capsules=(ca, cb),
            spec_id=spec_id,
        )
    specs = {
        ca.capsule_id: CapsuleSpec(holes={"a": [1]}, outputs=("a",)),
        cb.capsule_id: CapsuleSpec(holes={"b": [1]}, outputs=("b",)),
    }
    bounds = SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=10,
        max_backtracks=10,
        max_verifier_calls=10,
    )

    # Force the predecessor summary to be conservative.
    class ConservativeExtractor(FakeSummaryExtractor):
        def extract_summary(self, capsule, state):
            summary = super().extract_summary(capsule, state)
            return CapsuleInterfaceSummary(
                capsule_id=summary.capsule_id,
                input_bindings=summary.input_bindings,
                output_bindings=summary.output_bindings,
                slots=summary.slots,
                preconditions=summary.preconditions,
                postconditions=summary.postconditions,
                effects=summary.effects,
                exceptions=summary.exceptions,
                captures=summary.captures,
                conservative=True,
                fingerprint=summary.fingerprint,
            )

    result = solve_capsule_graph(
        graph,
        builder=FakeBuilder(specs),
        provider=_provider(specs, bounds),
        terminal_checker=FakeTerminalChecker(),
        summary_extractor=ConservativeExtractor(specs),
        materializer=make_materializer(),
        global_verifier=make_global_verifier(lambda _: False),
        bounds=bounds,
    )
    assert result.status == "unknown"
    assert result.stop_reason == "conservative_predecessor"
    assert result.capsule_results[1].status == "unknown"


def test_local_pass_global_fail_is_disagreement():
    spec_id = "disagree-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    nodes = (root, a)
    ca = _capsule(f"{spec_id}:ca", (a.node_id,))
    graph = _make_graph(
            root,
            nodes=nodes,
            edges=(),
            capsules=(ca,),
            spec_id=spec_id,
        )
    specs = {
        ca.capsule_id: CapsuleSpec(holes={"a": [1]}, outputs=("a",)),
    }
    bounds = SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=10,
        max_backtracks=10,
        max_verifier_calls=10,
    )
    result = solve_capsule_graph(
        graph,
        builder=FakeBuilder(specs),
        provider=_provider(specs, bounds),
        terminal_checker=FakeTerminalChecker(),
        summary_extractor=FakeSummaryExtractor(specs),
        materializer=make_materializer(),
        global_verifier=make_global_verifier(lambda src: src is not None and "ca=" in src),
        bounds=bounds,
    )
    assert result.status == "unknown"
    assert len(result.local_global_disagreements) == 1
    assert result.local_global_disagreements[0].kind == "local_pass_global_fail"


def test_missing_pack_hook_fails_closed():
    import re

    from slm_training.dsl.pack import CONTENT_PROPS, PLACEHOLDER_RE, PlaceholderPolicy, register_pack

    placeholder_policy = PlaceholderPolicy(
        placeholder_re=re.compile(PLACEHOLDER_RE.pattern),
        content_props=CONTENT_PROPS,
        slot_contract=lambda *_: (),
    )
    partial = DslPack(
        pack_id="partial-capsule-pack",
        backend=get_backend("openui"),
        placeholder_policy=placeholder_policy,
        reward_label="none",
    )
    register_pack(partial)
    with pytest.raises(PackSlotUnavailable):
        partial.require("capsule_problem_builder")


def test_summary_round_trips_json():
    summary = CapsuleInterfaceSummary(
        capsule_id="c1",
        input_bindings=(BindingSummary(name="x", kind="input"),),
        output_bindings=(BindingSummary(name="y", kind="output", value=_int_value(7)),),
        slots=(SlotSummary(name=":slot"),),
        preconditions=("p1",),
        postconditions=(),
        effects=(),
        exceptions=(),
        captures=(),
        conservative=False,
        fingerprint="a" * 64,
    )
    recovered = CapsuleInterfaceSummary.from_dict(summary.to_dict())
    assert recovered == summary
    assert recovered.is_exact


def test_result_round_trips_json():
    spec_id = "roundtrip-test"
    root = _root_node(spec_id)
    a = _statement_node(spec_id, "a")
    ca = _capsule(f"{spec_id}:ca", (a.node_id,))
    graph = _make_graph(
            root,
            nodes=(root, a),
            edges=(),
            capsules=(ca,),
            spec_id=spec_id,
        )
    specs = {
        ca.capsule_id: CapsuleSpec(holes={"a": [1]}, outputs=("a",)),
    }
    bounds = SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=10,
        max_backtracks=10,
        max_verifier_calls=10,
    )
    result = solve_capsule_graph(
        graph,
        builder=FakeBuilder(specs),
        provider=_provider(specs, bounds),
        terminal_checker=FakeTerminalChecker(),
        summary_extractor=FakeSummaryExtractor(specs),
        materializer=make_materializer(),
        global_verifier=make_global_verifier(lambda _: False),
        bounds=bounds,
    )
    recovered = CapsuleSolveResult.from_dict(result.to_dict())
    assert recovered.status == result.status
    assert recovered.counters.capsule_count == result.counters.capsule_count


def test_capsule_solver_is_torch_free():
    root = Path(__file__).parents[2]
    code = """
import sys
assert "torch" not in sys.modules
from slm_training.dsl.solver import capsule_solver  # noqa: F401
assert "torch" not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)
