"""PCT-004 (SLM-440): pack-owned completion-artifact provider seam.

Generalizes ownership of the certified static completion artifact behind
``StaticCompletionArtifactProvider`` without rebuilding the artifact or
changing OpenUI's default byte/decision path — see
``docs/design/certified-completion-artifact-and-tps-target.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    ARTIFACT_SCHEMA,
    CHECKER_SCHEMA,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_MANIFEST_PATH,
    REQUEST_DEPENDENT_AUTHORITY_EXCLUDED,
    provider_for_pack,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.pack import (
    completion_artifact_provider_for_grammar_path,
    get_pack,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def test_provider_for_openui_pack_resolves_the_committed_default_paths() -> None:
    provider = provider_for_pack(get_pack("openui"))

    assert provider is not None
    assert provider.pack_id == "openui"
    assert provider.artifact_path == DEFAULT_ARTIFACT_PATH
    assert provider.manifest_path == DEFAULT_MANIFEST_PATH


def test_provider_for_pack_without_completion_artifact_is_none() -> None:
    assert provider_for_pack(get_pack("toy-layout")) is None


def test_provider_describes_identity_authority_and_fallback() -> None:
    provider = provider_for_pack(get_pack("openui"))
    assert provider is not None

    description = provider.describe()

    assert description["pack_id"] == "openui"
    assert description["artifact_schema"] == ARTIFACT_SCHEMA
    assert description["checker_schema"] == CHECKER_SCHEMA
    assert description["safe_fallback"] == "exact_live_construction"
    assert set(REQUEST_DEPENDENT_AUTHORITY_EXCLUDED) <= set(
        description["request_dependent_authority_excluded"]
    )
    assert description["static_authority_covered"]


def test_grammar_path_lookup_generalizes_beyond_a_hardcoded_filename() -> None:
    openui_pack = get_pack("openui")
    toy_pack = get_pack("toy-layout")

    resolved = completion_artifact_provider_for_grammar_path(
        openui_pack.backend.info.grammar_path
    )
    assert resolved is not None
    assert resolved == provider_for_pack(openui_pack)

    # A pack that never declared an artifact yields no provider — the engine
    # must fall through to exact live construction, never widen support.
    assert (
        completion_artifact_provider_for_grammar_path(
            toy_pack.backend.info.grammar_path
        )
        is None
    )
    assert completion_artifact_provider_for_grammar_path(Path("/no/such/grammar.lark")) is None


def test_engine_direct_map_routes_through_the_pack_provider_seam() -> None:
    """Adapting OpenUI to the seam is byte/decision-identical to the old
    hardcoded-filename gate: same hit, same fallback counters, same map."""

    from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map

    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()

    mapping = engine._direct_map(tokenizer)

    assert mapping == dsl_direct_terminal_map(tokenizer, engine._parser.terminals)
    assert engine.stats["static_artifact_hits"] == 1
    assert engine.stats["static_artifact_fallbacks"] == 0


def test_provider_try_load_fails_closed_on_a_corrupt_artifact(tmp_path: Path) -> None:
    from slm_training.dsl.grammar.fastpath.completion_artifact import (
        StaticCompletionArtifactProvider,
    )

    artifact = tmp_path / "completion.safetensors"
    manifest = tmp_path / "completion.manifest.json"
    artifact.write_bytes(DEFAULT_ARTIFACT_PATH.read_bytes() + b"x")
    manifest.write_bytes(DEFAULT_MANIFEST_PATH.read_bytes())
    provider = StaticCompletionArtifactProvider(
        pack_id="openui", artifact_path=artifact, manifest_path=manifest
    )
    engine = OpenUIIncrementalEngine()

    checked, error = provider.try_load(
        DSLNativeTokenizer.build(), engine._parser, grammar_path=engine.grammar_path
    )

    assert checked is None
    assert error is not None and "artifact_sha256 mismatch" in error
    with pytest.raises(ValueError, match="artifact_sha256 mismatch"):
        provider.load_checked(
            DSLNativeTokenizer.build(),
            engine._parser,
            grammar_path=engine.grammar_path,
        )


def test_pack_without_artifact_still_gets_exact_live_construction() -> None:
    """A partial pack (no ``completion_artifact``) must never widen support —
    its engine keeps producing the exact live direct-feed map."""

    from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
    from slm_training.dsl.grammar.fastpath.token_map import dsl_direct_terminal_map

    engine = OpenUIIncrementalEngine(GRAMMARS_DIR / "toy_layout.lark")
    tokenizer = DSLNativeTokenizer.build()

    mapping = engine._direct_map(tokenizer)

    assert mapping == dsl_direct_terminal_map(tokenizer, engine._parser.terminals)
    assert engine.stats["static_artifact_hits"] == 0
    assert engine.stats["static_artifact_fallbacks"] == 0
