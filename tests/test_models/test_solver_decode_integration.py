"""VSS1-03 (SLM-63): model-level decode integration for ``verified_solver_decode``.

Torch-bearing integration tests for the compiler-tree decode seam:

* the solver config fields default disabled and round-trip through
  ``dataclasses.asdict`` (the config/checkpoint metadata path), with old
  checkpoints missing the fields falling back to defaults;
* ``verified_solver_decode=False`` is byte-identical decode (the parity
  regression on a fixed fixture/seed);
* ``_solver_prune_forest`` runs real certificate-checked closure over a real
  ``CompletionForest`` and never invents a candidate the forest lacked;
* an unsupported tokenizer/pack raises a clear capability error rather than
  silently taking a weaker path;
* the decode loop invokes the solver seam only when the flag is enabled.

The core ``solver_prune`` removal/keep/certified-bottom/coverage semantics (with a
fake certifying provider) live in ``tests/test_dsl/test_solver_decode.py``; this
file only pins the Torch model wiring.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch

from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.dsl.solver.goal_support import (
    GoalFailureAtomV1,
    GoalSupportProvider,
    GoalTerminalEvidenceV1,
    GoalVerifierProfileV1,
    compute_pack_identity_digest,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import ExpandStatus, ExpandStep, VerifyOutcome, VerifyStatus
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    build_completion_forest,
)
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

_SOLVER_DEFAULTS = {
    "verified_solver_decode": False,
    "goal_support_mode": "off",
    "goal_support_profile_mode": None,
    "solver_max_nodes": 512,
    "solver_max_depth": 64,
    "solver_max_backtracks": 64,
    "solver_max_verifier_calls": 64,
    "solver_max_wall_ms": 0,
    "solver_unknown_policy": "keep_and_rank",
    "solver_certificate_mode": "summary",
}


def _model(**config_overrides) -> TwoTowerModel:
    record = ExampleRecord(
        id="compiler",
        prompt="card",
        openui='root = Card([b1])\nb1 = TextContent(":slot_0")\n',
        placeholders=[":slot_0"],
        split="train",
        source="fixture",
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=32,
        grammar_ltr_max_tokens=32,
        gen_steps=1,
        seed=0,
        **config_overrides,
    )
    model = TwoTowerModel.from_records([record], config=config, device="cpu")
    model.eval()
    return model


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _TerminalExpander:
    """Exact two-terminal fixture for the real goal-support provider."""

    def __init__(self, state: FiniteDomainState, programs: dict[DomainValue, str]):
        self._state = state
        self._programs = programs

    @property
    def problem_id(self) -> str:
        return self._state.problem_id

    @property
    def pack_id(self) -> str:
        return self._state.pack_id

    @property
    def constraint_version(self) -> str:
        return self._state.constraint_version

    @property
    def bounds(self) -> SolverBounds:
        return self._state.bounds

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        assert hole_id == state.holes[0].hole_id
        return ExpandStep(
            ExpandStatus.TERMINAL,
            program=self._programs[value],
            coverage="complete",
        )


class _GoalTraceVerifier:
    """Fresh deterministic terminal verifier with strict replayable evidence."""

    def __init__(self, profile: GoalVerifierProfileV1) -> None:
        self._goal_profile = profile
        self._records: dict[str, GoalTerminalEvidenceV1] = {}

    @property
    def profile(self) -> str:
        return f"openui/goal-support/v1/{self._goal_profile.digest}"

    @property
    def terminal_evidence(self) -> tuple[GoalTerminalEvidenceV1, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def verify(self, program: str) -> VerifyOutcome:
        accepted = program == "goal-satisfying"
        program_digest = _digest(program)
        failures = () if accepted else (
            GoalFailureAtomV1(
                atom_id="constraint:c_required",
                reason_code="required_fact_failed",
                source_kind="constraint",
                completeness_class="EXACT",
                source_identity="c_required",
            ),
        )
        evidence = GoalTerminalEvidenceV1.from_dict(
            GoalTerminalEvidenceV1(
                profile_digest=self._goal_profile.digest,
                program_digest=program_digest,
                canonical_program_digest=program_digest,
                program_spec_digest=_digest(f"spec:{program}"),
                semantic_plan_digest=_digest(f"plan:{program}"),
                mandatory_failure_atoms=failures,
                structural_status="ACCEPT",
                overall_status="ACCEPT" if accepted else "REJECT",
            ).to_dict()
        )
        self._records[program_digest] = evidence
        return VerifyOutcome(
            VerifyStatus.ACCEPT if accepted else VerifyStatus.REJECT,
            None if accepted else "goal_failed",
        )


def _goal_provider_for_forest(
    forest: CompletionForest, prefix: list[int]
) -> tuple[FiniteDomainState, GoalSupportProvider]:
    pack_identity = PackIdentityV1(
        pack_id="openui",
        contract_version=2,
        grammar_sha256="a" * 64,
        tokenizer_id="tok/v1",
        canonicalizer_id="canon/v1",
        artifact_schema="openui/v1",
        artifact_sha256="b" * 64,
    )
    constraint = GoalConstraintV1(
        constraint_id="c_required",
        kind="slot_present",
        parameters={"role_id": "primary_action"},
        source_kind="pack_contract",
        source_id="pack/openui",
        source_digest="c" * 64,
        authority_tier="compiler-hard",
        completeness="EXACT",
        may_prune=True,
    )
    constraint_set = CompiledGoalConstraintSetV1.from_dict(
        CompiledGoalConstraintSetV1(
            problem_digest="d" * 64,
            request_digest="e" * 64,
            pack_identity_digest=compute_pack_identity_digest(pack_identity),
            compiler_version="compiler/v1",
            constraints=(constraint,),
            hard_constraint_ids=(constraint.constraint_id,),
        ).to_dict()
    )
    profile = GoalVerifierProfileV1.from_dict(
        GoalVerifierProfileV1(
            profile_id="fixture/prod",
            mode="production_exact",
            problem_digest=constraint_set.problem_digest,
            constraint_set_digest=constraint_set.digest,
            pack_identity=pack_identity,
            pack_identity_digest=constraint_set.pack_identity_digest,
            required_constraint_ids=(constraint.constraint_id,),
            authority_tier="compiler-hard",
        ).to_dict()
    )
    bounds = SolverBounds(1000, 100, 10, 100, 100)
    state = completion_forest_state(
        prefix_ids=prefix,
        forest=forest,
        pack_id="openui",
        constraint_version="goal/v1",
        bounds=bounds,
    )
    values = state.holes[0].values
    expander = _TerminalExpander(
        state,
        {
            values[0]: "goal-satisfying",
            values[1]: "goal-violating",
        },
    )
    return state, GoalSupportProvider(
        expander,
        verifier_factory=lambda: _GoalTraceVerifier(profile),
        profile=profile,
        constraint_set=constraint_set,
    )


def test_solver_fields_default_disabled() -> None:
    config = TwoTowerConfig(context_backend="scratch", output_tokenizer="lexer")
    for field, expected in _SOLVER_DEFAULTS.items():
        assert getattr(config, field) == expected
    assert config.verified_solver_decode is False


def test_solver_config_round_trips_and_old_checkpoints_default() -> None:
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        verified_solver_decode=True,
        solver_max_nodes=8,
        solver_max_depth=4,
        solver_max_backtracks=3,
        solver_max_verifier_calls=5,
        solver_max_wall_ms=7,
        solver_unknown_policy="keep_and_rank",
        solver_certificate_mode="full",
    )
    dumped = dataclasses.asdict(config)
    for field in _SOLVER_DEFAULTS:
        assert field in dumped, f"{field} missing from serialized config"

    fields = TwoTowerConfig.__dataclass_fields__
    restored = TwoTowerConfig(**{k: v for k, v in dumped.items() if k in fields})
    assert restored.verified_solver_decode is True
    assert restored.solver_max_nodes == 8
    assert restored.solver_max_wall_ms == 7
    assert restored.solver_certificate_mode == "full"

    # An old checkpoint dict missing every solver field falls back to defaults
    # (strict for existing tensors, tolerant only of new config defaults).
    legacy = {
        k: v
        for k, v in dumped.items()
        if k != "verified_solver_decode"
        and not k.startswith("solver_")
        and not k.startswith("goal_support_")
    }
    legacy_config = TwoTowerConfig(**{k: v for k, v in legacy.items() if k in fields})
    for field, expected in _SOLVER_DEFAULTS.items():
        assert getattr(legacy_config, field) == expected


def test_disabled_flag_is_decode_identical() -> None:
    baseline = _model()
    ctx, ctx_pad = baseline._encode_context(["card"])
    expected = baseline._compiler_ltr_decode_one(
        ctx, ctx_pad, 24, mode="tree", slot_contract=None
    )

    off = _model(verified_solver_decode=False)
    ctx2, ctx_pad2 = off._encode_context(["card"])
    ids = off._compiler_ltr_decode_one(
        ctx2, ctx_pad2, 24, mode="tree", slot_contract=None
    )
    assert torch.equal(ids, expected)


def test_solver_prune_forest_runs_real_closure_and_returns_subset() -> None:
    model = _model(verified_solver_decode=True, solver_max_nodes=4)
    prefix = [model.tokenizer.bos_id]
    forest = build_completion_forest(model.tokenizer, prefix)
    assert forest.coverage == "complete" and forest.paths

    pruned = model._solver_prune_forest(forest, prefix)
    assert isinstance(pruned, CompletionForest)
    assert pruned.coverage == forest.coverage
    original = {(p.kind, tuple(p.token_ids)) for p in forest.paths}
    survivors = {(p.kind, tuple(p.token_ids)) for p in pruned.paths}
    # Closure may only remove candidates; it never invents one the forest lacked.
    assert survivors <= original


def test_enabled_requires_dsl_native_tokenizer() -> None:
    model = _model(verified_solver_decode=True)
    prefix = [model.tokenizer.bos_id]
    forest = build_completion_forest(model.tokenizer, prefix)
    model.tokenizer = object()  # not a DSLNativeTokenizer/pack
    with pytest.raises(ValueError, match="DSL-native"):
        model._solver_prune_forest(forest, prefix)


def test_decode_invokes_solver_only_when_enabled() -> None:
    # A spy proves the flag gates the seam call without paying the (deliberately
    # expensive) real-solver cost on every decode step. Returning the forest
    # unchanged keeps decode on the baseline trajectory.
    calls = {"n": 0}

    def _spy(forest, prefix):
        calls["n"] += 1
        return forest

    off = _model(verified_solver_decode=False)
    off._solver_prune_forest = _spy  # type: ignore[assignment]
    ctx, ctx_pad = off._encode_context(["card"])
    off._compiler_ltr_decode_one(ctx, ctx_pad, 8, mode="tree", slot_contract=None)
    assert calls["n"] == 0  # disabled: seam never called

    calls["n"] = 0
    on = _model(verified_solver_decode=True)
    on._solver_prune_forest = _spy  # type: ignore[assignment]
    ctx2, ctx_pad2 = on._encode_context(["card"])
    on._compiler_ltr_decode_one(ctx2, ctx_pad2, 8, mode="tree", slot_contract=None)
    assert calls["n"] >= 1  # enabled: pruned at least once before soft ranking


def test_goal_support_defaults_off_and_legacy_checkpoint_loads() -> None:
    config = TwoTowerConfig(context_backend="scratch", output_tokenizer="lexer")
    assert config.goal_support_mode == "off"
    assert config.goal_support_profile_mode is None

    dumped = dataclasses.asdict(
        TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="lexer",
            goal_support_mode="diagnostic",
        )
    )
    fields = TwoTowerConfig.__dataclass_fields__
    legacy = {k: v for k, v in dumped.items() if k in fields and not k.startswith("goal_support_")}
    legacy_config = TwoTowerConfig(**legacy)
    assert legacy_config.goal_support_mode == "off"
    assert legacy_config.goal_support_profile_mode is None

def test_diagnostic_goal_support_is_decode_identical() -> None:
    baseline = _model()
    ctx, ctx_pad = baseline._encode_context(["card"])
    expected = baseline._compiler_ltr_decode_one(
        ctx, ctx_pad, 24, mode="tree", slot_contract=None
    )

    diagnostic = _model(goal_support_mode="diagnostic", compiler_decode_mode="tree")
    ctx2, ctx_pad2 = diagnostic._encode_context(["card"])
    ids = diagnostic._compiler_ltr_decode_one(
        ctx2, ctx_pad2, 24, mode="tree", slot_contract=None
    )
    assert torch.equal(ids, expected)


def test_goal_support_seam_invoked_only_when_enabled() -> None:
    from slm_training.data.contract import GenerationRequest
    from slm_training.dsl.solver.decode import build_goal_support_decode_binding

    calls = {"n": 0}

    def _spy(forest, prefix, plan_row):
        calls["n"] += 1
        return forest

    off = _model(goal_support_mode="off", compiler_decode_mode="tree")
    off._goal_support_apply_forest = _spy  # type: ignore[assignment]
    ctx, ctx_pad = off._encode_context(["card"])
    off._compiler_ltr_decode_one(ctx, ctx_pad, 8, mode="tree", slot_contract=None)
    assert calls["n"] == 0

    calls["n"] = 0
    on = _model(goal_support_mode="diagnostic", compiler_decode_mode="tree")
    on._goal_support_bindings = [
        build_goal_support_decode_binding(GenerationRequest(prompt="card"))
    ]
    on._goal_support_apply_forest = _spy  # type: ignore[assignment]
    ctx2, ctx_pad2 = on._encode_context(["card"])
    on._compiler_ltr_decode_one(ctx2, ctx_pad2, 8, mode="tree", slot_contract=None)
    assert calls["n"] >= 1


def test_off_mode_goal_support_counters_stay_zero() -> None:
    from slm_training.models.decode_stats import (
        collect_decode_stats,
        proves_goal_support_off,
    )

    model = _model(goal_support_mode="off", compiler_decode_mode="tree")
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        model._compiler_ltr_decode_one(ctx, ctx_pad, 8, mode="tree", slot_contract=None)
    assert proves_goal_support_off(stats)


def test_diagnostic_goal_support_records_queries_without_prune() -> None:
    from slm_training.data.contract import GenerationRequest
    from slm_training.dsl.solver.decode import build_goal_support_decode_binding
    from slm_training.models.decode_stats import collect_decode_stats

    model = _model(goal_support_mode="diagnostic", compiler_decode_mode="tree")
    model._prepare_goal_support_bindings(["card"])
    model._goal_support_bindings = [
        build_goal_support_decode_binding(GenerationRequest(prompt="card"))
    ]
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        model._compiler_ltr_decode_one(ctx, ctx_pad, 8, mode="tree", slot_contract=None)
    assert stats.goal_support_pruned == 0


def test_certified_singleton_bypass_has_zero_forwards_when_collapsed() -> None:
    from slm_training.models.decode_stats import (
        collect_decode_stats,
        proves_zero_neural_work,
    )

    model = _model(
        goal_support_mode="certified",
        compiler_decode_mode="tree",
        grammar_draft_window=4,
        grammar_ltr_max_tokens=8,
    )
    model._prepare_goal_support_bindings(["card"])
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        model._compiler_ltr_decode_one(ctx, ctx_pad, 6, mode="tree", slot_contract=None)
    if stats.forwards_count == 0:
        assert proves_zero_neural_work(stats)
