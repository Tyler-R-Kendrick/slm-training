"""OpenFeature-compatible experiment lever evaluation (SLM-342)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.flags import (
    FlagClient,
    InMemoryProvider,
    Reason,
    apply_experiment_flags,
    client_from_environ,
    evaluate_ofrep,
    experiment_context,
    ruleset_from_mapping,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.experiment_flags import (
    apply_levers_from_mapping,
    cli_lever_overrides,
)


def _config(**kwargs) -> ModelBuildConfig:
    base = dict(
        train_dir=Path("outputs/data/train/v1"),
        run_id="flag-test-run",
        model_name="twotower",
        context_backend="scratch",
    )
    base.update(kwargs)
    return ModelBuildConfig(**base)


def test_missing_ruleset_keeps_defaults_byte_identical() -> None:
    cfg = _config()
    before = (
        cfg.verified_solver_decode,
        cfg.honest_slot_contract,
        cfg.compiler_decode_mode,
        cfg.solver_max_nodes,
    )
    client = FlagClient(InMemoryProvider({}))  # empty ruleset
    cfg2, applied = apply_experiment_flags(
        cfg,
        client=client,
        context=experiment_context(run_id=cfg.run_id, experiment_id="E0"),
    )
    assert applied == []
    assert (
        cfg2.verified_solver_decode,
        cfg2.honest_slot_contract,
        cfg2.compiler_decode_mode,
        cfg2.solver_max_nodes,
    ) == before
    assert before == (False, False, "off", 512)


def test_ruleset_enables_verified_solver_decode() -> None:
    cfg = _config()
    cfg, applied = apply_levers_from_mapping(
        cfg,
        {"verified_solver_decode": True},
        experiment_id="E-vss",
        matrix="quality",
    )
    assert cfg.verified_solver_decode is True
    assert len(applied) == 1
    assert applied[0].flag_key == "verified_solver_decode"
    assert applied[0].variant == "on"
    assert applied[0].reason is Reason.STATIC
    assert applied[0].flag_metadata["experiment_id"] == "E-vss"


def test_cli_overrides_win_over_ruleset() -> None:
    cfg = _config(verified_solver_decode=False)
    client = FlagClient(
        InMemoryProvider(ruleset_from_mapping({"verified_solver_decode": True}))
    )
    cfg, applied = apply_experiment_flags(
        cfg,
        client=client,
        overrides={"verified_solver_decode": False},
    )
    assert cfg.verified_solver_decode is False
    assert applied[0].flag_metadata["source"] == "override"


def test_ofrep_evaluate_payload() -> None:
    client = FlagClient(
        InMemoryProvider(ruleset_from_mapping({"verified_solver_decode": True}))
    )
    payload = evaluate_ofrep(
        client,
        context_payload={"targetingKey": "run-1", "experiment_id": "E53"},
        flags=["verified_solver_decode", "honest_slot_contract"],
    )
    assert payload["flags"]["verified_solver_decode"]["value"] is True
    assert payload["flags"]["verified_solver_decode"]["variant"] == "on"
    # honest_slot_contract absent from ruleset → default false
    assert payload["flags"]["honest_slot_contract"]["value"] is False
    assert payload["flags"]["honest_slot_contract"]["reason"] == "DEFAULT"


def test_client_from_environ_json(tmp_path: Path) -> None:
    env = {"OPENUI_FLAGS_JSON": json.dumps({"asap_decode": True})}
    client = client_from_environ(environ=env)
    assert client is not None
    assert client.get_boolean_value("asap_decode", False) is True

    path = tmp_path / "flags.json"
    path.write_text(json.dumps({"solver_max_nodes": 128}), encoding="utf-8")
    client2 = client_from_environ(environ={"OPENUI_FLAGS_PATH": str(path)})
    assert client2 is not None
    assert client2.get_number_value("solver_max_nodes", 512) == 128


def test_cli_lever_overrides_store_true() -> None:
    class Args:
        verified_solver_decode = True
        honest_slot_contract = False
        ship_gates = False
        compiler_decode_mode = "off"
        asap_decode = False
        solver_max_nodes = 256
        solver_unknown_policy = "keep_and_rank"
        solver_certificate_mode = "summary"

    overrides = cli_lever_overrides(
        Args(), argv=("--verified-solver-decode", "--solver-max-nodes", "256")
    )
    assert overrides["verified_solver_decode"] is True
    assert overrides["solver_max_nodes"] == 256
    assert "compiler_decode_mode" not in overrides


def test_cli_lever_overrides_do_not_treat_argparse_defaults_as_explicit() -> None:
    class Args:
        verified_solver_decode = False
        honest_slot_contract = False
        ship_gates = False
        compiler_decode_mode = "off"
        asap_decode = False
        solver_max_nodes = 512
        solver_unknown_policy = "keep_and_rank"
        solver_certificate_mode = "summary"

    assert cli_lever_overrides(Args(), argv=()) == {}


def test_compiler_prefill_max_states_lever_default_matches_config_default() -> None:
    # Diagnostic lever added for the prefill-batch-cap experiment
    # (docs/design/lever-compiler-prefill-max-states-cap-exposure12-20260728.md):
    # the registry default (0=auto) must equal ModelBuildConfig's own field
    # default so an empty ruleset changes nothing (byte-identical defaults).
    cfg = _config()
    assert cfg.compiler_prefill_max_states == 0
    client = FlagClient(InMemoryProvider({}))  # empty ruleset
    cfg2, applied = apply_experiment_flags(
        cfg,
        client=client,
        context=experiment_context(run_id=cfg.run_id, experiment_id="E0"),
    )
    assert applied == []
    assert cfg2.compiler_prefill_max_states == 0


def test_compiler_prefill_max_states_lever_overridable() -> None:
    cfg = _config()
    cfg, applied = apply_levers_from_mapping(
        cfg,
        {"compiler_prefill_max_states": 1},
        experiment_id="E-prefill-cap",
        matrix="decode_latency",
    )
    assert cfg.compiler_prefill_max_states == 1
    assert any(a.flag_key == "compiler_prefill_max_states" for a in applied)


def test_unknown_lever_rejected() -> None:
    try:
        ruleset_from_mapping({"not_a_real_lever": True})
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "not_a_real_lever" in str(exc)
