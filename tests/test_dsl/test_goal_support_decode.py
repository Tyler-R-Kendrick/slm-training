"""PGS-E01 (SLM-504): goal-support decode seam on ``dsl/solver/decode.py``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest, CompletionPath
from slm_training.dsl.solver.decode import (
    GOAL_SUPPORT_MODES,
    build_goal_support_decode_binding,
    goal_support_certified_prune,
    goal_support_diagnostic_observe,
    normalize_goal_support_mode,
    solver_prune,
    validate_goal_support_config,
)
from slm_training.dsl.solver.state import SolverBounds, SupportVerdict
from tests.test_dsl.test_goal_support import _profile, _provider


_BOUNDS = SolverBounds(
    max_tokens=1000,
    max_nodes=1000,
    max_depth=32,
    max_backtracks=1000,
    max_verifier_calls=1000,
)


def _forest(*, coverage: str = "complete") -> CompletionForest:
    return CompletionForest(
        (
            CompletionPath((10,), "a"),
            CompletionPath((20,), "b"),
            CompletionPath((30,), "c"),
        ),
        coverage,
        ("NAME", "COMPONENT"),
    )


def test_normalize_goal_support_mode_defaults_off() -> None:
    assert normalize_goal_support_mode(None) == "off"
    assert normalize_goal_support_mode("") == "off"
    assert normalize_goal_support_mode("diagnostic") == "diagnostic"
    assert GOAL_SUPPORT_MODES == ("off", "diagnostic", "certified")
    with pytest.raises(ValueError, match="unsupported goal_support_mode"):
        normalize_goal_support_mode("bogus")


def test_validate_goal_support_config_rejects_certified_without_compiler_tree() -> None:
    config = SimpleNamespace(
        goal_support_mode="certified",
        compiler_decode_mode="off",
        goal_support_profile_mode=None,
    )
    with pytest.raises(ValueError, match="compiler_decode_mode != off"):
        validate_goal_support_config(config)


def test_build_goal_support_decode_binding_compiles_structured_request() -> None:
    binding = build_goal_support_decode_binding(
        GenerationRequest(prompt="Build a card")
    )
    assert binding.ready is True
    assert binding.profile is not None
    assert binding.profile.mode == "production_exact"
    assert binding.constraint_set is not None


def test_goal_support_diagnostic_observe_preserves_forest_identity() -> None:
    expander, provider = _provider({"ax"})
    forest = _forest()
    observed, diag = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
        state=expander.root_state(),
    )
    assert observed is forest
    assert [path.kind for path in observed.paths] == ["a", "b", "c"]
    assert diag.observed is True


def test_goal_support_diagnostic_skips_incomplete_forest() -> None:
    expander, provider = _provider({"ax"})
    forest = _forest(coverage="partial")
    observed, diag = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
        state=expander.root_state(),
    )
    assert observed is forest
    assert diag.observed is False
    assert diag.reason == "incomplete_forest"


def test_goal_support_certified_skips_incomplete_forest() -> None:
    expander, provider = _provider({"ax"})
    forest = _forest(coverage="partial")
    pruned, result = goal_support_certified_prune(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=expander.root_state(),
    )
    assert pruned is forest
    assert result is None


def test_goal_support_certified_runs_authority_guard() -> None:
    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "complete", ())
    pruned, result = goal_support_certified_prune(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    assert result is not None
    assert len(pruned.paths) <= len(forest.paths)


def test_goal_support_certified_rejects_non_goal_provider() -> None:
    from tests.test_dsl.test_solver_decode import _RuleProvider

    with pytest.raises(ValueError, match="GoalSupportProvider"):
        goal_support_certified_prune(
            _forest(),
            [1],
            _RuleProvider(lambda _tids: SupportVerdict.UNKNOWN),
            pack_id="fixture",
            constraint_version="cv",
            bounds=_BOUNDS,
        )


def test_goal_support_certified_rejects_evaluation_oracle_profile() -> None:
    expander, provider = _provider(
        {"ax"},
        profile=_profile(mode="evaluation_oracle", authority_tier="evaluation-only"),
    )
    with pytest.raises(ValueError, match="rejects non-production"):
        goal_support_certified_prune(
            _forest(),
            [1],
            provider,
            pack_id="fixture",
            constraint_version="cv",
            bounds=_BOUNDS,
            state=expander.root_state(),
        )


def test_solver_prune_still_uses_exact_closure_for_generic_provider() -> None:
    from tests.test_dsl.test_solver_decode import _RuleProvider

    forest = _forest()
    pruned, result = solver_prune(
        forest,
        [1],
        _RuleProvider(lambda tids: SupportVerdict.UNSUPPORTED if tids == (10,) else SupportVerdict.UNKNOWN),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    assert result is not None
    assert [path.kind for path in pruned.paths] == ["b", "c"]
