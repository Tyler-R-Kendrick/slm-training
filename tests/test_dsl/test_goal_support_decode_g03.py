"""PGS-G03 (SLM-509): adversarial decode parity, singleton bypass, legacy compatibility.

Proves goal-support modes cannot alter default/diagnostic behavior, weaken constraints,
bypass canonical pruning, or break historical artifacts after PGS-E01/E02.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from slm_training.dsl.solver import decode as goal_support_decode
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.solver.decode import (
    GOAL_SUPPORT_MODES,
    goal_support_certified_prune,
    goal_support_diagnostic_observe,
    normalize_goal_support_mode,
    solver_prune,
    validate_goal_support_config,
)
from slm_training.dsl.solver.state import SolverBounds, SupportVerdict
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import _twotower_config_from_build
from slm_training.levers import (
    CONSTRAINT_WEAKENING_LEVERS,
    constraint_weakening_violations,
    require_constrained_production_config,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    collect_decode_stats,
    fold_goal_support_closure,
    record_goal_support_incomplete_forest_skip,
)
from slm_training.models.twotower import TwoTowerConfig
from tests.test_dsl.test_goal_support import _profile, _provider
from tests.test_dsl.test_solver_decode import _RuleProvider, _forest, _kinds

_BOUNDS = SolverBounds(
    max_tokens=1000,
    max_nodes=1000,
    max_depth=32,
    max_backtracks=1000,
    max_verifier_calls=1000,
)

_HISTORICAL_TWOTOWER_DEFAULTS = {
    "context_backend": "scratch",
    "output_tokenizer": "lexer",
    "d_model": 32,
    "n_heads": 2,
    "context_layers": 1,
    "denoiser_layers": 1,
    "max_prompt_len": 32,
    "max_target_len": 32,
    "grammar_ltr_max_tokens": 32,
    "gen_steps": 1,
    "seed": 0,
    "compiler_decode_mode": "tree",
    "verified_solver_decode": False,
}


def _path_keys(forest: CompletionForest) -> list[tuple[str, tuple[int, ...]]]:
    return [(path.kind, tuple(path.token_ids)) for path in forest.paths]


def _assert_forest_parity(left: CompletionForest, right: CompletionForest) -> None:
    assert left.coverage == right.coverage
    assert left.terminals == right.terminals
    assert _path_keys(left) == _path_keys(right)


def _diagnostic_vs_off_forest(
    *,
    coverage: str = "complete",
    supported: set[str] | None = None,
) -> tuple[CompletionForest, CompletionForest]:
    expander, provider = _provider(supported or {"ax", "ay"})
    state = expander.root_state()
    if coverage == "complete" and supported is None:
        paths = tuple(
            CompletionPath((index,), value.payload["letter"])
            for index, value in enumerate(state.holes[0].values)
        )
        forest = CompletionForest(paths, coverage, ())
    else:
        forest = _forest(coverage)
    off = forest
    observed, _diag = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    return off, observed


# --------------------------------------------------------------------------- #
# Default/off compatibility
# --------------------------------------------------------------------------- #


def test_historical_twotower_config_without_goal_support_fields_defaults_off() -> None:
    config = TwoTowerConfig(**_HISTORICAL_TWOTOWER_DEFAULTS)
    assert config.goal_support_mode == "off"
    assert config.goal_support_profile_mode is None


def test_historical_checkpoint_dict_missing_goal_support_round_trips_off() -> None:
    config = TwoTowerConfig(
        **_HISTORICAL_TWOTOWER_DEFAULTS,
        goal_support_mode="diagnostic",
        goal_support_profile_mode="production_exact",
    )
    dumped = dataclasses.asdict(config)
    fields = TwoTowerConfig.__dataclass_fields__
    legacy = {
        k: v
        for k, v in dumped.items()
        if k in fields and not k.startswith("goal_support_")
    }
    restored = TwoTowerConfig(**legacy)
    assert restored.goal_support_mode == "off"
    assert restored.goal_support_profile_mode is None


def test_model_build_config_and_factory_default_goal_support_off() -> None:
    build = ModelBuildConfig(train_dir=".", output_tokenizer="lexer")
    assert build.goal_support_mode == "off"
    runtime = _twotower_config_from_build(build)
    assert runtime.goal_support_mode == "off"
    assert runtime.goal_support_profile_mode is None


def test_unknown_future_goal_support_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported goal_support_mode"):
        normalize_goal_support_mode("future_mode_v99")
    with pytest.raises(ValueError, match="unsupported goal_support_mode"):
        normalize_goal_support_mode("experimental")


@pytest.mark.parametrize(
    "raw",
    ["", "none", "false", "0", None],
)
def test_falsy_goal_support_modes_normalize_to_off(raw: str | None) -> None:
    assert normalize_goal_support_mode(raw) == "off"


def test_goal_support_mode_not_a_constraint_weakening_lever() -> None:
    assert "goal_support_mode" not in CONSTRAINT_WEAKENING_LEVERS


# --------------------------------------------------------------------------- #
# Diagnostic observational parity
# --------------------------------------------------------------------------- #


def test_diagnostic_preserves_forest_membership_and_order_vs_off() -> None:
    off, diagnostic = _diagnostic_vs_off_forest()
    _assert_forest_parity(off, diagnostic)


def test_diagnostic_repeat_is_deterministic() -> None:
    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "complete", ())
    first, diag_a = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    second, diag_b = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    _assert_forest_parity(first, second)
    assert diag_a.observed == diag_b.observed
    assert diag_a.reason == diag_b.reason


def test_diagnostic_closure_failure_cannot_affect_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expander, provider = _provider({"ax"})
    forest = _forest()
    state = expander.root_state()

    def _boom(*_a, **_k):
        raise RuntimeError("diagnostic-only failure")

    monkeypatch.setattr(
        "slm_training.dsl.solver.goal_support.exact_goal_closure",
        _boom,
    )
    observed, diag = goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    assert observed is forest
    assert diag.observed is False
    assert diag.reason is not None and "closure_failed" in diag.reason


def test_diagnostic_incomplete_forest_unchanged_and_counted() -> None:
    for coverage in ("partial", "none"):
        off, diagnostic = _diagnostic_vs_off_forest(coverage=coverage)
        _assert_forest_parity(off, diagnostic)
        stats = DecodeStats()
        record_goal_support_incomplete_forest_skip(stats, "diagnostic")
        assert stats.goal_support_skipped_incomplete_forest == 1


def test_diagnostic_folds_work_counters_without_prune() -> None:
    from tests.test_models.test_decode_stats import _closure_result

    stats = DecodeStats()
    fold_goal_support_closure(
        stats,
        _closure_result(support_queries=3, candidates_removed=2),
        mode="diagnostic",
    )
    assert stats.goal_support_queries == 3
    assert stats.goal_support_pruned == 0


def test_off_mode_never_invokes_certified_prune_semantics() -> None:
    forest = _forest()
    pruned, result = solver_prune(
        forest,
        [1],
        _RuleProvider(lambda tids: SupportVerdict.UNSUPPORTED if tids == (10,) else SupportVerdict.UNKNOWN),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    # Off path: solver_prune is separate; diagnostic never calls this via observe.
    assert _kinds(pruned) == ["b", "c"]
    assert result is not None


# --------------------------------------------------------------------------- #
# Certified authority / closure guard
# --------------------------------------------------------------------------- #


def test_certified_rejects_advisory_profile_at_config_boundary() -> None:
    config = SimpleNamespace(
        goal_support_mode="certified",
        compiler_decode_mode="tree",
        goal_support_profile_mode="advisory_diagnostic",
    )
    with pytest.raises(ValueError, match="forbidden|production_exact"):
        validate_goal_support_config(config)


def test_certified_rejects_evaluation_oracle_at_config_boundary() -> None:
    config = SimpleNamespace(
        goal_support_mode="certified",
        compiler_decode_mode="tree",
        goal_support_profile_mode="evaluation_oracle",
    )
    with pytest.raises(ValueError, match="forbidden|production_exact"):
        validate_goal_support_config(config)


def test_certified_skips_incomplete_and_none_forests() -> None:
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    for coverage in ("partial", "none"):
        forest = _forest(coverage)
        pruned, result = goal_support_certified_prune(
            forest,
            [1],
            provider,
            pack_id=expander.pack_id,
            constraint_version=expander.constraint_version,
            bounds=expander.bounds,
            state=state,
        )
        assert pruned is forest
        assert result is None


def test_certified_rejects_non_goal_support_provider() -> None:
    with pytest.raises(ValueError, match="GoalSupportProvider"):
        goal_support_certified_prune(
            _forest(),
            [1],
            _RuleProvider(lambda _t: SupportVerdict.UNKNOWN),
            pack_id="fixture",
            constraint_version="cv",
            bounds=_BOUNDS,
        )


def test_certified_rejects_evaluation_oracle_provider() -> None:
    expander, provider = _provider(
        {"ax"},
        profile=_profile(mode="evaluation_oracle", authority_tier="evaluation-only"),
    )
    with pytest.raises(ValueError, match="rejects non-production"):
        goal_support_certified_prune(
            _forest(),
            [1],
            provider,
            pack_id=expander.pack_id,
            constraint_version=expander.constraint_version,
            bounds=expander.bounds,
            state=expander.root_state(),
        )


def test_replay_failure_never_authorizes_prune() -> None:
    from slm_training.dsl.solver.support import ReplayResult

    expander, provider = _provider({"ax"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "complete", ())
    real_replay = provider.replay

    def _bad_replay(certificate, *, state):
        return ReplayResult(ok=False, verdict=certificate.verdict, violations=("replay",))

    provider.replay = _bad_replay  # type: ignore[method-assign]
    pruned, result = goal_support_certified_prune(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    provider.replay = real_replay  # type: ignore[method-assign]
    assert len(pruned.paths) == len(forest.paths)
    assert result is not None


# --------------------------------------------------------------------------- #
# Constraint safety
# --------------------------------------------------------------------------- #


def test_certified_bottom_yields_empty_forest_not_fallback() -> None:
    forest = _forest()
    with pytest.raises(ValueError, match="GoalSupportProvider"):
        goal_support_certified_prune(
            forest,
            [1],
            _RuleProvider(lambda _t: SupportVerdict.UNSUPPORTED),
            pack_id="fixture",
            constraint_version="cv",
            bounds=_BOUNDS,
        )
    pruned, result = solver_prune(
        forest,
        [1],
        _RuleProvider(lambda _t: SupportVerdict.UNSUPPORTED),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    assert pruned.paths == ()
    assert result is not None and result.state.is_bottom


def test_prune_never_adds_candidates_only_subset() -> None:
    forest = _forest()
    pruned, _result = solver_prune(
        forest,
        [1],
        _RuleProvider(lambda tids: SupportVerdict.UNSUPPORTED if tids == (10,) else SupportVerdict.UNKNOWN),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    original = {(p.kind, tuple(p.token_ids)) for p in forest.paths}
    survivors = {(p.kind, tuple(p.token_ids)) for p in pruned.paths}
    assert survivors <= original
    assert len(survivors) < len(original)


def test_production_config_with_goal_support_still_blocks_unconstrained_fallback() -> None:
    config = SimpleNamespace(
        grammar_constrained=True,
        allow_unconstrained_fallback=True,
        grammar_fastpath=True,
        goal_support_mode="certified",
    )
    violations = constraint_weakening_violations(config)
    assert "allow_unconstrained_fallback" in violations
    with pytest.raises(ValueError, match="allow_unconstrained_fallback"):
        require_constrained_production_config(config)


def test_goal_support_modes_are_not_weakening_lever_values() -> None:
    for mode in GOAL_SUPPORT_MODES:
        config = SimpleNamespace(
            grammar_constrained=True,
            allow_unconstrained_fallback=False,
            grammar_fastpath=True,
            goal_support_mode=mode,
        )
        assert constraint_weakening_violations(config) == {}


# --------------------------------------------------------------------------- #
# Certified singleton (torch-free closure seam)
# --------------------------------------------------------------------------- #


def test_certified_closure_collapses_to_singleton_candidate() -> None:
    forest = _forest()
    pruned, result = solver_prune(
        forest,
        [1],
        _RuleProvider(
            lambda tids: SupportVerdict.UNSUPPORTED
            if tids in ((10,), (20,))
            else SupportVerdict.UNKNOWN
        ),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    assert result is not None
    assert len(pruned.paths) == 1
    assert pruned.coverage == "complete"
    assert pruned.paths[0].kind == "c"


# --------------------------------------------------------------------------- #
# Empty and unknown domains
# --------------------------------------------------------------------------- #


def test_unknown_candidates_remain_live_and_rankable() -> None:
    forest = _forest()
    pruned, result = solver_prune(
        forest,
        [1],
        _RuleProvider(lambda _t: SupportVerdict.UNKNOWN),
        pack_id="fixture",
        constraint_version="cv",
        bounds=_BOUNDS,
    )
    assert pruned is forest
    assert result is not None


def test_budget_exhaustion_never_upgrades_unknown_to_unsupported() -> None:
    from slm_training.dsl.solver.closure import exact_closure

    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    result = exact_closure(state, provider, max_queries=0)
    assert result.stop_reason == "budget:max_queries"
    assert result.counters.support_queries == 0


def test_timeout_bound_exhaustion_keeps_candidates() -> None:
    from slm_training.dsl.solver.closure import exact_closure

    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    before = len(state.holes[0].values)
    result = exact_closure(state, provider, max_queries=0)
    after = len(result.state.holes[0].values)
    assert result.stop_reason == "budget:max_queries"
    assert after == before


# --------------------------------------------------------------------------- #
# Mutation guards — tests must fail when bypass/identity breaks
# --------------------------------------------------------------------------- #


def test_mutation_diagnostic_identity_return_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    expander, provider = _provider({"ax", "ay"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "complete", ())

    def _evil_mutate(forest_in, *_a, **_k):
        trimmed = CompletionForest(forest_in.paths[:1], forest_in.coverage, forest_in.terminals)
        return trimmed, SimpleNamespace(observed=True, reason=None, closure_result=None)

    monkeypatch.setattr(
        goal_support_decode,
        "goal_support_diagnostic_observe",
        _evil_mutate,
    )
    observed, _diag = goal_support_decode.goal_support_diagnostic_observe(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    with pytest.raises(AssertionError):
        _assert_forest_parity(forest, observed)


def test_mutation_certified_must_not_preserve_full_forest_on_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expander, provider = _provider({"ax"})
    state = expander.root_state()
    paths = tuple(
        CompletionPath((index,), value.payload["letter"])
        for index, value in enumerate(state.holes[0].values)
    )
    forest = CompletionForest(paths, "complete", ())

    def _noop(_forest, *_a, **_k):
        return _forest, None

    monkeypatch.setattr(
        "slm_training.dsl.solver.decode.goal_support_certified_prune",
        _noop,
    )
    pruned, result = goal_support_certified_prune(
        forest,
        [1],
        provider,
        pack_id=expander.pack_id,
        constraint_version=expander.constraint_version,
        bounds=expander.bounds,
        state=state,
    )
    with pytest.raises(AssertionError):
        assert len(pruned.paths) < len(forest.paths)
        assert result is not None


# --------------------------------------------------------------------------- #
# Torch-bearing singleton bypass (isolated; never run full integration file)
# --------------------------------------------------------------------------- #


def test_certified_singleton_bypass_zero_forwards_with_spies(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.models.decode_stats import proves_zero_neural_work

    record = ExampleRecord(
        id="g03",
        prompt="card",
        openui='root = Card([b1])\nb1 = TextContent(":slot_0")\n',
        placeholders=[":slot_0"],
        split="train",
        source="fixture",
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="lexer",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=32,
            grammar_ltr_max_tokens=8,
            gen_steps=1,
            seed=0,
            compiler_decode_mode="tree",
            goal_support_mode="certified",
            grammar_draft_window=4,
        ),
        device="cpu",
    )
    model.eval()
    model._prepare_goal_support_bindings(["card"])

    singleton = CompletionForest(
        (CompletionPath((model.tokenizer.eos_id,), "eos"),),
        "complete",
        (),
    )
    calls = {
        "forward": 0,
        "select": 0,
        "solver_energy": 0,
        "speculative": 0,
    }

    monkeypatch.setattr(
        compiler_draft,
        "build_completion_forest",
        lambda *_a, **_k: CompletionForest(
            (
                CompletionPath((model.tokenizer.eos_id,), "eos"),
                CompletionPath((model.tokenizer.bos_id + 1,), "alt"),
            ),
            "complete",
            (),
        ),
    )

    def _certified_singleton(forest, prefix, plan_row):
        return singleton

    monkeypatch.setattr(model, "_goal_support_apply_forest", _certified_singleton)
    monkeypatch.setattr(
        model,
        "_denoiser_forward",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("forwarded")),
    )
    monkeypatch.setattr(
        model,
        "_denoiser_hidden",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("hidden")),
    )

    if hasattr(model, "_rank_solver_energy"):
        monkeypatch.setattr(
            model,
            "_rank_solver_energy",
            lambda *_a, **_k: calls.__setitem__("solver_energy", calls["solver_energy"] + 1),
        )
    if hasattr(model, "_speculative_rank_paths"):
        monkeypatch.setattr(
            model,
            "_speculative_rank_paths",
            lambda *_a, **_k: calls.__setitem__("speculative", calls["speculative"] + 1),
        )

    original_select = model._select_compiler_path

    def _counting_select(*args, **kwargs):
        calls["select"] += 1
        return original_select(*args, **kwargs)

    monkeypatch.setattr(model, "_select_compiler_path", _counting_select)

    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        ids = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 6, mode="tree", slot_contract=None
        )
    assert ids is not None
    assert stats.forwards_count == 0
    assert proves_zero_neural_work(stats)
    assert calls["forward"] == 0
    assert calls["solver_energy"] == 0
    assert calls["speculative"] == 0
    assert stats.goal_support_pruned >= 0
    assert stats.forced_row_tokens_without_forward >= 1 or stats.forced_tokens >= 1
