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

import pytest
import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    build_completion_forest,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

_SOLVER_DEFAULTS = {
    "verified_solver_decode": False,
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
        if k != "verified_solver_decode" and not k.startswith("solver_")
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
