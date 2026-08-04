from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_MANIFEST_PATH,
    _decode_safetensors,
    _encode_safetensors,
    _sha256_bytes,
    build_completion_artifact,
    completion_artifact_checkpoint_identity,
    load_checked_completion_artifact,
    require_checkpoint_completion_artifact,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def test_committed_artifact_certifies_live_static_authority() -> None:
    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()

    checked = load_checked_completion_artifact(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )

    assert checked.direct_map == dsl_direct_terminal_map(
        tokenizer, engine._parser.terminals
    )
    assert checked.manifest["lalr_state_count"] > 0
    assert checked.manifest["lalr_edge_count"] > checked.manifest["lalr_state_count"]
    assert (
        "scope visibility" in checked.manifest["request_dependent_authority_excluded"]
    )
    identity = completion_artifact_checkpoint_identity()
    assert identity["sha256"] == checked.digest
    assert identity["checker_schema"].endswith("/v1")
    require_checkpoint_completion_artifact({"completion_artifact": identity})
    with pytest.raises(ValueError, match="does not match"):
        require_checkpoint_completion_artifact(
            {"completion_artifact": {**identity, "sha256": "0" * 64}}
        )


def test_engine_uses_checked_artifact_without_weakening_fallback() -> None:
    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()

    assert engine._direct_map(tokenizer) is not None
    assert engine.stats["static_artifact_hits"] == 1
    assert engine.stats["static_artifact_fallbacks"] == 0


def test_checker_rejects_corrupt_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "completion.safetensors"
    manifest = tmp_path / "completion.manifest.json"
    artifact.write_bytes(DEFAULT_ARTIFACT_PATH.read_bytes() + b"x")
    manifest.write_bytes(DEFAULT_MANIFEST_PATH.read_bytes())
    engine = OpenUIIncrementalEngine()

    with pytest.raises(ValueError, match="artifact_sha256 mismatch"):
        load_checked_completion_artifact(
            DSLNativeTokenizer.build(),
            engine._parser,
            grammar_path=engine.grammar_path,
            artifact_path=artifact,
            manifest_path=manifest,
        )


def test_committed_artifact_carries_certified_rule_and_bound_tensors() -> None:
    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()
    checked = load_checked_completion_artifact(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    tensors, _metadata = _decode_safetensors(DEFAULT_ARTIFACT_PATH.read_bytes())
    rules = list(engine._parser.rules)

    assert len(tensors["rule_origins"]) == len(rules)
    assert len(tensors["rule_lengths"]) == len(rules)
    assert checked.manifest["lalr_rule_count"] == len(rules)
    assert len(tensors["state_min_terminals"]) == checked.manifest["lalr_state_count"]
    bound = checked.manifest["state_min_terminals"]
    assert bound["kind"] == "control_only_lower_bound"
    assert bound["direction"] == "negative_only"
    assert bound["consumed_by"] == []
    assert bound["trivial"] is (bound["max"] <= 0)
    assert "LOWER bound" in checked.manifest["proof_obligations"]["state_min_terminals"]
    assert checked.lalr is not None
    assert checked.lalr.start_stack() == (
        checked.manifest["control"]["start_states"]["start"],
    )


@pytest.mark.parametrize(
    "tensor", ["rule_origins", "rule_lengths", "state_min_terminals"]
)
def test_checker_rejects_tampered_control_tensor(tmp_path: Path, tensor: str) -> None:
    """Re-derivation, not just the digest, catches a rewritten tensor."""

    raw = DEFAULT_ARTIFACT_PATH.read_bytes()
    tensors, metadata = _decode_safetensors(raw)
    assert _encode_safetensors(tensors, metadata) == raw
    tensors[tensor] = [value + 1 for value in tensors[tensor]]
    tampered = _encode_safetensors(tensors, metadata)
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    # Keep every digest honest so the failure can only come from the deep check.
    manifest["artifact_sha256"] = _sha256_bytes(tampered)
    artifact_path = tmp_path / "tampered.safetensors"
    manifest_path = tmp_path / "tampered.manifest.json"
    artifact_path.write_bytes(tampered)
    manifest_path.write_text(json.dumps(manifest))
    engine = OpenUIIncrementalEngine()

    with pytest.raises(ValueError, match=f"control table mismatch for {tensor}"):
        load_checked_completion_artifact(
            DSLNativeTokenizer.build(),
            engine._parser,
            grammar_path=engine.grammar_path,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )


def test_builder_is_byte_deterministic(tmp_path: Path) -> None:
    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()
    first_artifact = tmp_path / "first.safetensors"
    first_manifest = tmp_path / "first.manifest.json"
    second_artifact = tmp_path / "second.safetensors"
    second_manifest = tmp_path / "second.manifest.json"

    first = build_completion_artifact(
        tokenizer,
        engine._parser,
        grammar_path=engine.grammar_path,
        artifact_path=first_artifact,
        manifest_path=first_manifest,
    )
    second = build_completion_artifact(
        tokenizer,
        engine._parser,
        grammar_path=engine.grammar_path,
        artifact_path=second_artifact,
        manifest_path=second_manifest,
    )

    assert first_artifact.read_bytes() == second_artifact.read_bytes()
    # The file name is intentionally descriptive metadata; certified content
    # and every authority digest remain identical.
    first.pop("artifact")
    second.pop("artifact")
    assert first == second
    assert json.loads(first_manifest.read_text())["checker_schema"].endswith("/v1")
