from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_MANIFEST_PATH,
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
