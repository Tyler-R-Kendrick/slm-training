from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.solver import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
    completion_forest_state,
)

BOUNDS = SolverBounds(64, 32, 8, 16, 20)


def value(number: int, *, tag: str = "token") -> DomainValue:
    return DomainValue.create(tag, {"id": number})


def hole(name: str, values: tuple[DomainValue, ...]) -> HoleDomain:
    return HoleDomain(
        HoleId("test", (name, 0), "choice"),
        values,
        (("support_verdict", SupportVerdict.UNKNOWN.value), ("coverage", "complete")),
    )


def state(*holes: HoleDomain) -> FiniteDomainState:
    return FiniteDomainState("problem", "openui", "v1", BOUNDS, tuple(holes))


def test_canonical_order_and_json_round_trip_stabilize_fingerprint() -> None:
    a = hole("a", (value(2), value(1)))
    b = HoleDomain(
        HoleId("test", (1, "mixed"), "choice"),
        (value(4), value(3)),
        (("z", 1), ("a", True)),
    )
    first = state(a, b)
    second = state(
        HoleDomain(b.hole_id, tuple(reversed(b.values)), tuple(reversed(b.metadata))),
        HoleDomain(a.hole_id, tuple(reversed(a.values)), tuple(reversed(a.metadata))),
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    payload = json.loads(json.dumps(first.to_dict()))
    restored = FiniteDomainState.from_dict(payload)
    assert restored == first
    assert restored.fingerprint == first.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "changed",
    [
        lambda base: replace(base, problem_id="other"),
        lambda base: replace(base, pack_id="other"),
        lambda base: replace(base, constraint_version="v2"),
        lambda base: replace(base, bounds=replace(BOUNDS, max_nodes=33)),
        lambda base: replace(
            base,
            holes=(hole("other", (value(1), value(2))),),
        ),
        lambda base: replace(base, holes=(hole("a", (value(1),)),)),
        lambda base: replace(
            base,
            holes=(
                HoleDomain(
                    base.holes[0].hole_id,
                    base.holes[0].values,
                    (("coverage", "partial"),),
                ),
            ),
        ),
    ],
)
def test_fingerprint_changes_for_each_hard_field(changed) -> None:
    base = state(hole("a", (value(1), value(2))))
    assert changed(base).fingerprint != base.fingerprint


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "max_nodes",
        "max_depth",
        "max_backtracks",
        "max_verifier_calls",
    ],
)
def test_fingerprint_changes_for_each_bound(field: str) -> None:
    base = state(hole("a", (value(1), value(2))))
    changed_bounds = replace(BOUNDS, **{field: getattr(BOUNDS, field) + 1})
    assert replace(base, bounds=changed_bounds).fingerprint != base.fingerprint


def test_fingerprint_excludes_reversible_search_lineage() -> None:
    base = state(hole("a", (value(1),)))
    lineage = replace(base, decision_level=3, parent_fingerprint="a" * 64)
    assert lineage.fingerprint == base.fingerprint


def test_bottom_structurally_solved_and_summary_semantics() -> None:
    bottom = state(hole("empty", ()))
    solved = state(hole("one", (value(1),)))
    open_state = state(hole("many", (value(1), value(2))))

    assert bottom.is_bottom and not bottom.is_structurally_solved
    assert solved.is_structurally_solved and not solved.is_bottom
    assert not open_state.is_bottom and not open_state.is_structurally_solved
    assert open_state.summary() == {
        "hole_count": 1,
        "unresolved_count": 1,
        "total_candidate_count": 2,
        "max_domain_size": 2,
        "mean_domain_size": 2.0,
        "is_bottom": False,
        "is_structurally_solved": False,
    }


def test_refine_is_monotone_and_unknown_holes_fail_with_identity() -> None:
    one, two, three = value(1), value(2), value(3)
    base = state(hole("a", (one, two)))
    hole_id = base.holes[0].hole_id

    refined = base.refine(hole_id, (two,), certificate_ref="future-proof-ref")
    assert refined.domain(hole_id).values == (two,)
    assert base.domain(hole_id).values == (one, two)
    assert base.refine(hole_id, ()).is_bottom
    with pytest.raises(ValueError, match="problem.*cannot add candidates"):
        base.refine(hole_id, (one, three))
    with pytest.raises(LookupError, match="problem.*no hole"):
        base.refine(HoleId("test", ("missing",), "choice"), ())


def test_meet_intersects_domains_and_rejects_mismatched_identity() -> None:
    one, two, three = value(1), value(2), value(3)
    left = state(hole("a", (one, two)))
    right = state(hole("a", (two, three)))
    met = left.meet(right)

    assert met.holes[0].values == (two,)
    assert met.is_structurally_solved
    with pytest.raises(ValueError, match="mismatched identity"):
        left.meet(replace(right, pack_id="other"))
    with pytest.raises(ValueError, match="mismatched holes"):
        left.meet(state(hole("b", (two,))))


@pytest.mark.parametrize(
    "changed",
    [
        lambda base: replace(base, problem_id="other"),
        lambda base: replace(base, pack_id="other"),
        lambda base: replace(base, constraint_version="v2"),
        lambda base: replace(base, bounds=replace(BOUNDS, max_tokens=65)),
        lambda base: replace(base, bounds=replace(BOUNDS, max_nodes=33)),
        lambda base: replace(base, bounds=replace(BOUNDS, max_depth=9)),
        lambda base: replace(base, bounds=replace(BOUNDS, max_backtracks=17)),
        lambda base: replace(base, bounds=replace(BOUNDS, max_verifier_calls=21)),
    ],
)
def test_meet_rejects_every_identity_mismatch(changed) -> None:
    base = state(hole("a", (value(1), value(2))))
    with pytest.raises(ValueError, match="mismatched identity"):
        base.meet(changed(base))


def test_meet_resets_unrelated_reversible_lineage() -> None:
    base = state(hole("a", (value(1), value(2))))
    left = replace(base, decision_level=2, parent_fingerprint="a" * 64)
    right = replace(base, decision_level=3, parent_fingerprint="b" * 64)
    met = left.meet(right)
    assert met.decision_level == 0
    assert met.parent_fingerprint is None


def test_decision_records_parent_without_claiming_support() -> None:
    one, two = value(1), value(2)
    base = state(hole("a", (one, two)))
    chosen = base.with_decision(base.holes[0].hole_id, two)

    assert chosen.decision_level == 1
    assert chosen.parent_fingerprint == base.fingerprint
    assert chosen.holes[0].values == (two,)
    assert dict(chosen.holes[0].metadata)["support_verdict"] == "unknown"


def test_validation_rejects_duplicates_invalid_bounds_and_non_json() -> None:
    one = value(1)
    with pytest.raises(ValueError, match="duplicate values"):
        hole("a", (one, one))
    with pytest.raises(ValueError, match="non-negative max_tokens"):
        replace(BOUNDS, max_tokens=-1)
    with pytest.raises(ValueError, match="not JSON-safe"):
        DomainValue.create("bad", object())
    with pytest.raises(ValueError, match="duplicate hole IDs"):
        state(hole("a", (one,)), hole("a", (value(2),)))
    with pytest.raises(ValueError, match="requires DomainValue candidates"):
        state(hole("a", (one,))).refine(hole("a", (one,)).hole_id, ({},))


@pytest.mark.parametrize("left,right", [(True, 1), (1, 1.0), (-0.0, 0.0)])
def test_metadata_json_types_remain_distinct(left, right) -> None:
    hole_id = HoleId("test", ("typed",), "choice")
    first = state(HoleDomain(hole_id, (value(1),), (("typed", left),)))
    second = state(HoleDomain(hole_id, (value(1),), (("typed", right),)))
    assert first != second
    assert first.fingerprint != second.fingerprint
    with pytest.raises(ValueError, match="mismatched metadata"):
        first.meet(second)


def test_completion_forest_adapter_preserves_full_paths_kind_and_coverage() -> None:
    forest = CompletionForest(
        (
            CompletionPath((11, 21), "component"),
            CompletionPath((11, 20), "component"),
            CompletionPath((11, 20), "binder"),
        ),
        "partial",
    )
    projected = completion_forest_state(
        prefix_ids=[1, 2],
        forest=forest,
        pack_id="openui",
        constraint_version="v1",
        bounds=BOUNDS,
    )
    payloads = [value.payload for value in projected.holes[0].values]

    assert payloads == [
        {"kind": "binder", "token_ids": [11, 20]},
        {"kind": "component", "token_ids": [11, 20]},
        {"kind": "component", "token_ids": [11, 21]},
    ]
    assert dict(projected.holes[0].metadata) == {
        "coverage": "partial",
        "support_verdict": "unknown",
    }
    assert projected.holes[0].hole_id.path[0] == 2
    assert not projected.is_structurally_solved


def test_completion_forest_adapter_empty_singleton_and_ordering() -> None:
    kwargs = {
        "prefix_ids": (1,),
        "pack_id": "openui",
        "constraint_version": "v1",
        "bounds": BOUNDS,
    }
    empty = completion_forest_state(forest=CompletionForest((), "none"), **kwargs)
    path_a = CompletionPath((10, 12), "component")
    path_b = CompletionPath((11,), "binder")
    singleton = completion_forest_state(
        forest=CompletionForest((path_a,), "complete"), **kwargs
    )
    ordered = completion_forest_state(
        forest=CompletionForest((path_a, path_b), "complete"), **kwargs
    )
    reversed_state = completion_forest_state(
        forest=CompletionForest((path_b, path_a), "complete"), **kwargs
    )

    assert empty.is_bottom
    assert singleton.is_structurally_solved
    assert dict(singleton.holes[0].metadata)["support_verdict"] == "unknown"
    assert ordered == reversed_state
    assert ordered.fingerprint == reversed_state.fingerprint
    with pytest.raises(ValueError, match="duplicate values"):
        completion_forest_state(
            forest=CompletionForest((path_a, path_a), "complete"), **kwargs
        )
    with pytest.raises(ValueError, match="prefix_ids must be non-negative integers"):
        completion_forest_state(
            forest=CompletionForest((path_a,), "complete"),
            **{**kwargs, "prefix_ids": (True,)},
        )


@pytest.mark.parametrize(
    ("forest", "message"),
    [
        (CompletionForest((), "bogus"), "coverage"),
        (CompletionForest((CompletionPath((1,), ""),), "complete"), "kind"),
        (
            CompletionForest((CompletionPath((True,), "component"),), "complete"),
            "token_ids",
        ),
        (
            CompletionForest((CompletionPath((-1,), "component"),), "complete"),
            "token_ids",
        ),
    ],
)
def test_completion_forest_adapter_rejects_malformed_input(
    forest: CompletionForest, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        completion_forest_state(
            prefix_ids=(),
            forest=forest,
            pack_id="openui",
            constraint_version="v1",
            bounds=BOUNDS,
        )


def test_completion_forest_adapter_ignores_explanation_only_fields() -> None:
    @dataclass(frozen=True)
    class ExtendedForest(CompletionForest):
        evidence: tuple[str, ...] = ()

    path = CompletionPath((10, 12), "component")
    kwargs = {
        "prefix_ids": (1,),
        "pack_id": "openui",
        "constraint_version": "v1",
        "bounds": BOUNDS,
    }
    first = completion_forest_state(
        forest=ExtendedForest((path,), "complete", evidence=("first",)), **kwargs
    )
    second = completion_forest_state(
        forest=ExtendedForest((path,), "complete", evidence=("second",)), **kwargs
    )
    assert first == second
    assert first.fingerprint == second.fingerprint


def test_solver_package_import_does_not_require_torch() -> None:
    root = Path(__file__).parents[2]
    code = """
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError(f'unexpected torch import: {fullname}')
        return None
sys.meta_path.insert(0, BlockTorch())
import slm_training.dsl.solver
assert 'torch' not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root, env=env)
