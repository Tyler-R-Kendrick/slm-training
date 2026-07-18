"""VSS3-03 (SLM-71): topology finite-domain solver integration tests.

These are wiring/regression tests for the guarded solver seam in
``GrammarDiffusionModel``.  They do not assert ship-quality parsing; the
solver mode is off by default and only exercised in fixture scale here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.grammar_diffusion import (
    GrammarDiffusionConfig,
    GrammarDiffusionModel,
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'

_TOPOLOGY_SOLVER_DEFAULTS = {
    "topology_verified_solver": False,
    "topology_capsule_solver": False,
    "topology_solver_ranker": "model",
    "topology_solver_unknown_policy": "keep_and_rank",
    "topology_solver_max_nodes": 256,
    "topology_solver_max_backtracks": 64,
    "topology_solver_max_verifier_calls": 64,
    "topology_solver_certificate_mode": "summary",
    "topology_solver_local_oracle": True,
    "topology_solver_global_verify": True,
}


def _model(**config_overrides) -> GrammarDiffusionModel:
    records = [
        ExampleRecord(
            id="hero",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":hero.title", ":hero.body"],
        ),
    ]
    config = GrammarDiffusionConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        topology_max_nodes=32,
        topology_max_active=8,
        topology_max_phases=8,
        seed=0,
        **config_overrides,
    )
    model = GrammarDiffusionModel.from_records(records, config=config, device="cpu")
    model.eval()
    return model


def test_topology_solver_fields_default_disabled() -> None:
    config = GrammarDiffusionConfig()
    for field, expected in _TOPOLOGY_SOLVER_DEFAULTS.items():
        assert getattr(config, field) == expected


def test_topology_solver_config_round_trips_and_old_checkpoints_default() -> None:
    config = GrammarDiffusionConfig(
        topology_verified_solver=True,
        topology_capsule_solver=False,
        topology_solver_ranker="deterministic",
        topology_solver_max_nodes=64,
        topology_solver_max_backtracks=8,
        topology_solver_max_verifier_calls=8,
        topology_solver_certificate_mode="full",
    )
    dumped = dataclasses.asdict(config)
    for field in _TOPOLOGY_SOLVER_DEFAULTS:
        assert field in dumped, f"{field} missing from serialized config"

    fields = GrammarDiffusionConfig.__dataclass_fields__
    restored = GrammarDiffusionConfig(
        **{k: v for k, v in dumped.items() if k in fields}
    )
    assert restored.topology_verified_solver is True
    assert restored.topology_solver_ranker == "deterministic"
    assert restored.topology_solver_max_nodes == 64

    legacy = {
        k: v
        for k, v in dumped.items()
        if not k.startswith("topology_solver_")
        and k not in {"topology_verified_solver", "topology_capsule_solver"}
    }
    legacy_config = GrammarDiffusionConfig(
        **{k: v for k, v in legacy.items() if k in fields}
    )
    for field, expected in _TOPOLOGY_SOLVER_DEFAULTS.items():
        assert getattr(legacy_config, field) == expected


def _request() -> GenerationRequest:
    return GenerationRequest(
        prompt="Hero", slot_contract=":hero.title :hero.body".split()
    )


def test_disabled_flag_is_decode_identical() -> None:
    model = _model()
    request = _request()
    baseline_text = model.generate_batch_requests([request])[0]
    baseline_stats = model.consume_generation_evidence()[0]

    model.config.topology_verified_solver = False
    off_text = model.generate_batch_requests([request])[0]
    off_stats = model.consume_generation_evidence()[0]

    assert off_text == baseline_text
    assert off_stats["topology_solver"] == {"enabled": False}
    assert off_stats["phases"] == baseline_stats["phases"]
    assert off_stats["expansions"] == baseline_stats["expansions"]


def test_enabled_invokes_solver_seam_and_records_trace() -> None:
    model = _model(
        topology_verified_solver=True,
        topology_solver_max_nodes=16,
        topology_solver_max_verifier_calls=8,
    )
    request = _request()
    text = model.generate_batch_requests([request])[0]
    stats = model.consume_generation_evidence()[0]
    trace = stats["topology_solver"]
    assert trace["enabled"] is True
    assert "support_queries" in trace
    assert "candidates_removed" in trace
    assert isinstance(text, str)


def test_enabled_solver_prune_is_monotone() -> None:
    """The solver seam is invoked at least once when enabled."""
    model = _model(
        topology_verified_solver=True,
        topology_solver_max_nodes=16,
        topology_solver_max_verifier_calls=8,
    )
    calls = {"n": 0}

    def _spy(root, slot_inventory, output_kind):
        calls["n"] += 1
        from slm_training.dsl.solver.topology_solver import (
            TopologyAdapterConfig,
            topology_solver_prune,
        )
        from slm_training.dsl.solver.state import SolverBounds

        adapter_config = TopologyAdapterConfig(
            topology_max_nodes=model.config.topology_max_nodes,
            topology_max_active=model.config.topology_max_active,
            topology_max_arity=model.config.topology_max_arity,
            topology_max_depth=model.config.topology_max_depth,
            topology_bounded_buffer=model.config.topology_bounded_buffer,
            topology_global_sync_interval=model.config.topology_global_sync_interval,
        )
        bounds = SolverBounds(
            max_tokens=model.config.topology_max_nodes * 64,
            max_nodes=model.config.topology_solver_max_nodes,
            max_depth=8,
            max_backtracks=model.config.topology_solver_max_backtracks,
            max_verifier_calls=model.config.topology_solver_max_verifier_calls,
        )
        return topology_solver_prune(
            root,
            model.codec,
            adapter_config,
            slot_inventory,
            output_kind,
            bounds,
            max_queries=4,
        )

    model._topology_solver_survivors = _spy  # type: ignore[assignment]
    request = _request()
    model.generate_batch_requests([request])
    assert calls["n"] >= 1


def test_capsule_solver_requires_verified_solver() -> None:
    records = [
        ExampleRecord(
            id="hero",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":hero.title", ":hero.body"],
        ),
    ]
    with pytest.raises(ValueError, match="topology_capsule_solver=True requires"):
        build_model(
            config=ModelBuildConfig(
                train_dir=Path("outputs/data/train/v1"),
                model_name="grammar_diffusion",
                topology_verified_solver=False,
                topology_capsule_solver=True,
            ),
            records=records,
        )
