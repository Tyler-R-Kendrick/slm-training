"""Tests for the Torch-free finite-domain support lattice (VSS0-03 / SLM-59).

Contract: ``docs/design/verified-scope-solver.md``. These tests exercise the
state invariants (order-insensitive fingerprints, monotone refinement, meet,
lineage, bottom/solved semantics, JSON round-trips), the completion-forest
adapter fixtures, and the Torch-free import guarantee.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest, CompletionPath
from slm_training.dsl.solver import (
    CompletionForestProjection,
    completion_forest_state,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainProjection,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
)

BOUNDS = SolverBounds(
    max_tokens=8, max_nodes=16, max_depth=4, max_backtracks=2, max_verifier_calls=3
)


def _value(token_ids: tuple[int, ...], kind: str = "component") -> DomainValue:
    return DomainValue(tag="token_path", token_ids=token_ids, kind=kind)


def _hole(
    path: tuple[str | int, ...],
    values: tuple[DomainValue, ...],
    metadata: tuple[tuple[str, object], ...] = (),
) -> HoleDomain:
    return HoleDomain(HoleId("prog", path, "next_action"), values, metadata)


def _state(*holes: HoleDomain, **overrides: object) -> FiniteDomainState:
    kwargs: dict[str, object] = {
        "problem_id": "prob-1",
        "pack_id": "openui",
        "constraint_version": "v1",
        "bounds": BOUNDS,
        "holes": holes,
    }
    kwargs.update(overrides)
    return FiniteDomainState(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Fingerprints.
# --------------------------------------------------------------------------- #
def test_fingerprint_is_order_insensitive() -> None:
    hole_a = _hole(("a",), (_value((1,)), _value((2,)), _value((3,))))
    hole_b = _hole(("b",), (_value((4,)),))
    forward = _state(hole_a, hole_b)
    reversed_holes = _state(hole_b, hole_a)
    assert forward.fingerprint == reversed_holes.fingerprint

    # Value order within a domain must not matter either.
    shuffled = _hole(("a",), (_value((3,)), _value((1,)), _value((2,))))
    assert _state(shuffled, hole_b).fingerprint == forward.fingerprint


def test_fingerprint_changes_for_every_hard_state_or_version_change() -> None:
    base = _state(_hole(("a",), (_value((1,)), _value((2,)))))
    fingerprint = base.fingerprint
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), problem_id="other").fingerprint != fingerprint
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), pack_id="other").fingerprint != fingerprint
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), constraint_version="v2").fingerprint != fingerprint
    other_bounds = SolverBounds(
        max_tokens=8, max_nodes=17, max_depth=4, max_backtracks=2, max_verifier_calls=3
    )
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), bounds=other_bounds).fingerprint != fingerprint
    # Domain content changes (value edit, removal, extra hole).
    assert _state(_hole(("a",), (_value((1,)), _value((9,))))).fingerprint != fingerprint
    assert _state(_hole(("a",), (_value((1,)),))).fingerprint != fingerprint
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), _hole(("b",), (_value((3,)),))).fingerprint != fingerprint


def test_fingerprint_excludes_search_trail_lineage() -> None:
    base = _state(_hole(("a",), (_value((1,)), _value((2,)))))
    assert _state(_hole(("a",), (_value((1,)), _value((2,)))), decision_level=5).fingerprint == base.fingerprint
    assert (
        _state(_hole(("a",), (_value((1,)), _value((2,)))), parent_fingerprint="deadbeef").fingerprint
        == base.fingerprint
    )


def test_fingerprint_stable_across_json_round_trip() -> None:
    state = _state(
        _hole(("a",), (_value((1, 2, 3)), _value((4,))), metadata=(("coverage", "partial"),)),
        _hole(("b", 0), (_value((7,), kind="bind"),)),
        decision_level=2,
        parent_fingerprint="cafef00d",
    )
    restored = FiniteDomainState.from_dict(state.to_dict())
    assert restored == state
    assert restored.fingerprint == state.fingerprint


# --------------------------------------------------------------------------- #
# Bottom / structurally solved.
# --------------------------------------------------------------------------- #
def test_bottom_semantics() -> None:
    bottom = _state(_hole(("a",), ()), _hole(("b",), (_value((1,)),)))
    assert bottom.is_bottom
    assert not bottom.is_structurally_solved


def test_structurally_solved_requires_all_singletons_and_not_bottom() -> None:
    solved = _state(_hole(("a",), (_value((1,)),)), _hole(("b",), (_value((2,)),)))
    assert solved.is_structurally_solved
    assert not solved.is_bottom
    unresolved = _state(_hole(("a",), (_value((1,)), _value((2,)))))
    assert not unresolved.is_structurally_solved
    # No holes: no remaining decisions -> vacuously structurally solved, not bottom.
    empty = _state()
    assert empty.is_structurally_solved
    assert not empty.is_bottom


# --------------------------------------------------------------------------- #
# Refinement (monotone) and decisions (lineage).
# --------------------------------------------------------------------------- #
def test_legal_monotone_refinement() -> None:
    hole_id = HoleId("prog", ("a",), "next_action")
    state = _state(HoleDomain(hole_id, (_value((1,)), _value((2,)), _value((3,)))))
    refined = state.refine(hole_id, (_value((1,)), _value((2,))))
    assert {v.canonical_key for v in refined.domain(hole_id).values} == {
        _value((1,)).canonical_key,
        _value((2,)).canonical_key,
    }
    # Refinement is not a decision: lineage is untouched.
    assert refined.decision_level == state.decision_level
    assert refined.parent_fingerprint == state.parent_fingerprint


def test_refine_rejects_candidate_expansion() -> None:
    hole_id = HoleId("prog", ("a",), "next_action")
    state = _state(HoleDomain(hole_id, (_value((1,)),)))
    with pytest.raises(ValueError, match="monotonicity violated"):
        state.refine(hole_id, (_value((1,)), _value((99,))))


def test_refine_rejects_unknown_hole() -> None:
    state = _state(_hole(("a",), (_value((1,)),)))
    with pytest.raises(ValueError, match="unknown hole"):
        state.refine(HoleId("prog", ("missing",), "next_action"), ())


def test_refine_certificate_ref_is_type_checked_and_fingerprint_neutral() -> None:
    hole_id = HoleId("prog", ("a",), "next_action")
    state = _state(HoleDomain(hole_id, (_value((1,)), _value((2,)))))
    with_cert = state.refine(hole_id, (_value((1,)),), certificate_ref="cert://abc")
    without_cert = state.refine(hole_id, (_value((1,)),))
    # A certificate reference is provenance only; it never changes state identity.
    assert with_cert.fingerprint == without_cert.fingerprint
    with pytest.raises(ValueError, match="certificate_ref"):
        state.refine(hole_id, (_value((1,)),), certificate_ref=123)  # type: ignore[arg-type]


def test_with_decision_records_lineage_without_claiming_proof() -> None:
    hole_id = HoleId("prog", ("a",), "next_action")
    state = _state(HoleDomain(hole_id, (_value((1,)), _value((2,)))))
    child = state.with_decision(hole_id, (_value((1,)),))
    assert child.decision_level == state.decision_level + 1
    assert child.parent_fingerprint == state.fingerprint
    # The domain narrowed, so the child's own fingerprint differs from its parent.
    assert child.fingerprint != state.fingerprint
    assert child.domain(hole_id).is_singleton


# --------------------------------------------------------------------------- #
# Meet.
# --------------------------------------------------------------------------- #
def test_meet_intersects_shared_holes_and_unions_disjoint_holes() -> None:
    left = _state(
        _hole(("a",), (_value((1,)), _value((2,)), _value((3,)))),
        _hole(("only_left",), (_value((5,)),)),
    )
    right = _state(
        _hole(("a",), (_value((2,)), _value((3,)), _value((4,)))),
        _hole(("only_right",), (_value((6,)),)),
    )
    met = left.meet(right)
    shared = met.domain(HoleId("prog", ("a",), "next_action"))
    assert {v.token_ids for v in shared.values} == {(2,), (3,)}
    assert met.has_hole(HoleId("prog", ("only_left",), "next_action"))
    assert met.has_hole(HoleId("prog", ("only_right",), "next_action"))
    # Commutative in identity terms.
    assert left.meet(right).fingerprint == right.meet(left).fingerprint


def test_meet_empty_intersection_yields_bottom() -> None:
    left = _state(_hole(("a",), (_value((1,)),)))
    right = _state(_hole(("a",), (_value((2,)),)))
    assert left.meet(right).is_bottom


def test_meet_rejects_mismatched_identity() -> None:
    left = _state(_hole(("a",), (_value((1,)),)))
    with pytest.raises(ValueError, match="problem_id"):
        left.meet(_state(_hole(("a",), (_value((1,)),)), problem_id="other"))
    with pytest.raises(ValueError, match="pack_id"):
        left.meet(_state(_hole(("a",), (_value((1,)),)), pack_id="other"))
    with pytest.raises(ValueError, match="constraint_version"):
        left.meet(_state(_hole(("a",), (_value((1,)),)), constraint_version="v2"))
    other_bounds = SolverBounds(
        max_tokens=99, max_nodes=16, max_depth=4, max_backtracks=2, max_verifier_calls=3
    )
    with pytest.raises(ValueError, match="bounds"):
        left.meet(_state(_hole(("a",), (_value((1,)),)), bounds=other_bounds))


# --------------------------------------------------------------------------- #
# Validation / rejection.
# --------------------------------------------------------------------------- #
def test_rejects_duplicate_hole_ids() -> None:
    with pytest.raises(ValueError, match="duplicate hole id"):
        _state(_hole(("a",), (_value((1,)),)), _hole(("a",), (_value((2,)),)))


def test_rejects_duplicate_domain_values() -> None:
    with pytest.raises(ValueError, match="duplicate domain value"):
        _hole(("a",), (_value((1,)), _value((1,))))


def test_rejects_negative_bounds() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SolverBounds(max_tokens=-1, max_nodes=1, max_depth=1, max_backtracks=1, max_verifier_calls=1)


def test_unknown_verdict_preserved_and_never_translated() -> None:
    assert SupportVerdict.UNKNOWN.value == "unknown"
    assert SupportVerdict.SUPPORTED.value == "supported"
    assert SupportVerdict.UNSUPPORTED.value == "unsupported"
    # Round-tripping through the str value never coerces UNKNOWN to UNSUPPORTED.
    assert SupportVerdict("unknown") is SupportVerdict.UNKNOWN


def test_summary_metrics() -> None:
    state = _state(
        _hole(("a",), (_value((1,)), _value((2,)))),
        _hole(("b",), (_value((3,)),)),
    )
    summary = state.summary()
    assert summary["hole_count"] == 2
    assert summary["unresolved_count"] == 1
    assert summary["total_candidates"] == 3
    assert summary["max_domain_size"] == 2
    assert summary["mean_domain_size"] == pytest.approx(1.5)
    assert summary["is_bottom"] is False
    assert summary["is_structurally_solved"] is False


# --------------------------------------------------------------------------- #
# Completion-forest adapter.
# --------------------------------------------------------------------------- #
def _project(forest: CompletionForest, prefix: tuple[int, ...] = (1, 2)) -> FiniteDomainState:
    return completion_forest_state(
        prefix_ids=prefix,
        forest=forest,
        pack_id="openui",
        constraint_version="v1",
        bounds=BOUNDS,
    )


def test_adapter_empty_forest_is_bottom() -> None:
    state = _project(CompletionForest((), "none"))
    assert state.is_bottom
    assert state.holes[0].metadata == (("coverage", "none"),)


def test_adapter_singleton_path_is_structurally_solved_for_projection() -> None:
    state = _project(CompletionForest((CompletionPath((5, 6, 7), "component"),), "complete"))
    assert state.is_structurally_solved
    assert state.holes[0].values[0].kind == "component"


def test_adapter_multi_path_forest_stays_unresolved() -> None:
    forest = CompletionForest(
        (CompletionPath((5,), "component"), CompletionPath((8,), "eos")), "complete"
    )
    state = _project(forest)
    assert not state.is_structurally_solved
    assert len(state.holes[0].values) == 2


def test_adapter_preserves_full_forced_suffix_not_just_first_token() -> None:
    forest = CompletionForest((CompletionPath((5, 6, 7), "component"),), "complete")
    state = _project(forest)
    assert state.holes[0].values[0].token_ids == (5, 6, 7)


def test_adapter_carries_partial_coverage_and_keeps_candidates_live() -> None:
    forest = CompletionForest(
        (CompletionPath((5,), "component"), CompletionPath((6,), "component_bound")), "partial"
    )
    state = _project(forest)
    assert state.holes[0].metadata == (("coverage", "partial"),)
    # Partial coverage never removes candidates at the projection layer.
    assert len(state.holes[0].values) == 2


def test_adapter_hole_is_keyed_by_prefix() -> None:
    forest = CompletionForest((CompletionPath((5,), "component"),), "complete")
    same = _project(forest, prefix=(1, 2))
    other = _project(forest, prefix=(1, 2, 3))
    assert same.fingerprint == _project(forest, prefix=(1, 2)).fingerprint
    assert same.holes[0].hole_id != other.holes[0].hole_id
    assert same.problem_id != other.problem_id


def test_adapter_state_round_trips() -> None:
    forest = CompletionForest(
        (CompletionPath((5, 6), "component"), CompletionPath((8,), "eos")), "partial"
    )
    state = _project(forest)
    assert FiniteDomainState.from_dict(state.to_dict()) == state


def test_completion_forest_projection_satisfies_protocol() -> None:
    forest = CompletionForest((CompletionPath((5,), "component"),), "complete")
    projection = CompletionForestProjection(
        prefix_ids=(1, 2), forest=forest, pack_id="openui", constraint_version="v1", bounds=BOUNDS
    )
    assert isinstance(projection, FiniteDomainProjection)
    assert projection.finite_domain_state() == _project(forest)


# --------------------------------------------------------------------------- #
# Torch-free import guarantee.
# --------------------------------------------------------------------------- #
def test_solver_package_imports_without_torch() -> None:
    import slm_training

    src_dir = str(Path(slm_training.__file__).resolve().parents[1])
    script = textwrap.dedent(
        """
        import sys
        import importlib.abc


        class _NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("No module named %r (blocked)" % name)
                return None


        sys.meta_path.insert(0, _NoTorch())
        import slm_training.dsl.solver as solver
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            CompletionForest,
            CompletionPath,
        )

        forest = CompletionForest((CompletionPath((5, 6, 7), "component"),), "complete")
        state = solver.completion_forest_state(
            prefix_ids=[1, 2],
            forest=forest,
            pack_id="openui",
            constraint_version="v1",
            bounds=solver.SolverBounds(8, 8, 8, 8, 8),
        )
        assert "torch" not in sys.modules, "torch was imported"
        assert state.is_structurally_solved
        assert state.holes[0].values[0].token_ids == (5, 6, 7)
        print("TORCH_FREE_OK")
        """
    )
    env = {**os.environ, "PYTHONPATH": src_dir}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "TORCH_FREE_OK" in result.stdout
