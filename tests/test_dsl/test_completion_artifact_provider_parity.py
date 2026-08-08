"""PCT-005 (SLM-451): prove byte/digest/decision/tamper parity through the
PCT-004 provider seam.

``StaticCompletionArtifactProvider.load_checked`` is a thin delegate to the
pre-existing ``load_checked_completion_artifact`` -- this file proves that
ownership refactor never changes a byte, a decision, or a failure mode, and
locks that parity in as a regression guard. ``test_completion_artifact.py``
already covers most tamper vectors through the *old* direct path;
``test_completion_artifact_provider.py`` covers provider resolution/fallback
wiring. This file is the differential/tamper matrix specifically *through*
the provider seam that PCT-005 asks for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath.completion_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_MANIFEST_PATH,
    StaticCompletionArtifactProvider,
    _decode_safetensors,
    _encode_safetensors,
    _sha256_bytes,
    completion_artifact_checkpoint_identity,
    load_checked_completion_artifact,
    provider_for_pack,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map
from slm_training.dsl.pack import get_pack
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def _engine_and_tokenizer() -> tuple[OpenUIIncrementalEngine, DSLNativeTokenizer]:
    return OpenUIIncrementalEngine(), DSLNativeTokenizer.build()


def _provider() -> StaticCompletionArtifactProvider:
    provider = provider_for_pack(get_pack("openui"))
    assert provider is not None
    return provider


def test_provider_load_checked_is_byte_digest_and_decision_identical_to_direct() -> None:
    engine, tokenizer = _engine_and_tokenizer()

    direct = load_checked_completion_artifact(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    via_provider = _provider().load_checked(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )

    assert direct.manifest == via_provider.manifest
    assert direct.direct_map == via_provider.direct_map
    assert direct.digest == via_provider.digest
    assert direct.lalr == via_provider.lalr


def test_provider_lalr_every_prefix_domain_matches_direct_path() -> None:
    """Structural adapter equality implies every-prefix accepts()/step() parity
    (both are pure functions of the adapter's own fields + the stack)."""
    engine, tokenizer = _engine_and_tokenizer()

    direct = load_checked_completion_artifact(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    via_provider = _provider().load_checked(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert direct.lalr is not None and via_provider.lalr is not None
    assert direct.lalr == via_provider.lalr

    stack = direct.lalr.start_stack()
    assert direct.lalr.accepts(stack) == via_provider.lalr.accepts(stack)


def test_provider_checkpoint_identity_matches_the_direct_checkpoint_identity() -> None:
    """Checkpoint artifact-binding behavior is unaffected by the seam (AC3)."""
    description = _provider().describe()
    identity = completion_artifact_checkpoint_identity()

    assert description["artifact_schema"] == identity["schema"]
    assert description["checker_schema"] == identity["checker_schema"]


def test_provider_payload_excludes_dynamic_runtime_facts() -> None:
    """Dynamic runtime/scope/semantic facts are absent from the static
    provider payload (AC4) -- only the documented exclusion label list may
    name them."""
    description = _provider().describe()
    dynamic_markers = ("runtime_symbol", "scope_state", "binder", "placeholder_value")
    flattened = json.dumps(description)
    for marker in dynamic_markers:
        assert marker not in flattened


def test_provider_describe_makes_no_cold_warm_performance_claim() -> None:
    """No claim of cold/warm improvement is made here (AC5)."""
    description = _provider().describe()
    flattened = " ".join(str(k) for k in description).lower()
    for banned in ("speedup", "throughput", "latency_ms", "cold_warm"):
        assert banned not in flattened


def _tampered_copy(
    tmp_path: Path, *, mutate_tensors=None, mutate_metadata=None, mutate_manifest=None
) -> tuple[Path, Path]:
    """Build a tampered (artifact, manifest) pair, keeping every digest that
    isn't the target of ``mutate_manifest`` internally consistent so a
    failure can only come from the specific check under test."""
    raw = DEFAULT_ARTIFACT_PATH.read_bytes()
    tensors, metadata = _decode_safetensors(raw)
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    if mutate_tensors is not None:
        mutate_tensors(tensors)
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    tampered = _encode_safetensors(tensors, metadata)
    manifest["artifact_sha256"] = _sha256_bytes(tampered)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    artifact_path = tmp_path / "tampered.safetensors"
    manifest_path = tmp_path / "tampered.manifest.json"
    artifact_path.write_bytes(tampered)
    manifest_path.write_text(json.dumps(manifest))
    return artifact_path, manifest_path


def test_provider_fails_closed_on_a_byte_flipped_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "flipped.safetensors"
    manifest = tmp_path / "flipped.manifest.json"
    artifact.write_bytes(DEFAULT_ARTIFACT_PATH.read_bytes() + b"x")
    manifest.write_bytes(DEFAULT_MANIFEST_PATH.read_bytes())
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact, manifest_path=manifest
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None and "artifact_sha256 mismatch" in error
    with pytest.raises(ValueError, match="artifact_sha256 mismatch"):
        provider.load_checked(tokenizer, engine._parser, grammar_path=engine.grammar_path)


def test_provider_fails_closed_on_a_stale_grammar_path(tmp_path: Path) -> None:
    """A grammar-path swap that changes grammar_sha256 must fail closed
    through the provider exactly as it would through the direct path."""
    provider = StaticCompletionArtifactProvider(
        pack_id="openui",
        artifact_path=DEFAULT_ARTIFACT_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
    )
    engine, tokenizer = _engine_and_tokenizer()
    stale_grammar_path = GRAMMARS_DIR / "toy_layout.lark"
    assert stale_grammar_path.is_file()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=stale_grammar_path
    )
    assert checked is None
    assert error is not None and "grammar_sha256 mismatch" in error
    with pytest.raises(ValueError, match="grammar_sha256 mismatch"):
        provider.load_checked(tokenizer, engine._parser, grammar_path=stale_grammar_path)


def test_provider_fails_closed_on_unsupported_manifest_schema(tmp_path: Path) -> None:
    artifact_path, manifest_path = _tampered_copy(
        tmp_path, mutate_manifest=lambda m: m.__setitem__("schema", "unsupported/v0")
    )
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact_path, manifest_path=manifest_path
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None and "unsupported completion artifact schema" in error


def test_provider_fails_closed_on_unsupported_artifact_metadata_schema(
    tmp_path: Path,
) -> None:
    artifact_path, manifest_path = _tampered_copy(
        tmp_path, mutate_metadata=lambda m: m.__setitem__("schema", "unsupported/v0")
    )
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact_path, manifest_path=manifest_path
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None and "artifact metadata schema mismatch" in error


def test_provider_fails_closed_on_tokenizer_row_shape_abuse(tmp_path: Path) -> None:
    """A truncated per-token tensor is a shape-abuse attack, not a legitimate
    tokenizer -- the provider must reject it exactly like the direct path."""

    def _truncate(tensors: dict[str, list[int]]) -> None:
        tensors["token_roles"] = tensors["token_roles"][:1]
        tensors["token_terminals"] = tensors["token_terminals"][:1]

    artifact_path, manifest_path = _tampered_copy(tmp_path, mutate_tensors=_truncate)
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact_path, manifest_path=manifest_path
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None and "artifact tokenizer row count mismatch" in error


@pytest.mark.parametrize(
    "tensor", ["rule_origins", "rule_lengths", "state_min_terminals"]
)
def test_provider_fails_closed_on_tampered_control_tensor(
    tmp_path: Path, tensor: str
) -> None:
    """Re-derivation, not just the digest, catches a rewritten control
    tensor -- proven through the provider seam (test_completion_artifact.py
    already proves this through the direct path)."""

    def _bump(tensors: dict[str, list[int]]) -> None:
        tensors[tensor] = [value + 1 for value in tensors[tensor]]

    artifact_path, manifest_path = _tampered_copy(tmp_path, mutate_tensors=_bump)
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact_path, manifest_path=manifest_path
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None and f"control table mismatch for {tensor}" in error
    with pytest.raises(ValueError, match=f"control table mismatch for {tensor}"):
        provider.load_checked(tokenizer, engine._parser, grammar_path=engine.grammar_path)


def test_failed_checker_never_uses_artifact_derived_authority(tmp_path: Path) -> None:
    """AC: failed checker cases never use artifact-derived authority -- a
    rejected load returns None, never a partial/degraded CompletionArtifact."""
    artifact = tmp_path / "flipped.safetensors"
    manifest = tmp_path / "flipped.manifest.json"
    artifact.write_bytes(DEFAULT_ARTIFACT_PATH.read_bytes() + b"x")
    manifest.write_bytes(DEFAULT_MANIFEST_PATH.read_bytes())
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact, manifest_path=manifest
    )
    engine, tokenizer = _engine_and_tokenizer()

    checked, error = provider.try_load(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    )
    assert checked is None
    assert error is not None
    # The safe fallback -- exact live construction -- remains available and
    # correct even while the artifact is rejected.
    fallback = dsl_direct_terminal_map(tokenizer, engine._parser.terminals)
    assert fallback is not None
    assert fallback == load_checked_completion_artifact(
        tokenizer, engine._parser, grammar_path=engine.grammar_path
    ).direct_map
