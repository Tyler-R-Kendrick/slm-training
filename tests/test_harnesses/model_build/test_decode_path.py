"""DecodePathSpec registry: required paths, compatibility, determinism."""

from __future__ import annotations

import pytest

from slm_training.harnesses.model_build.decode_path import (
    REQUIRED_DECODE_PATH_IDS,
    DecodePathSpec,
    all_decode_paths,
    compatible_decode_paths,
    get_decode_path,
)


def test_required_paths_present_in_order() -> None:
    ids = [p.path_id for p in all_decode_paths()]
    assert ids[:3] == list(REQUIRED_DECODE_PATH_IDS)


def test_fingerprint_deterministic_and_roundtrips() -> None:
    for spec in all_decode_paths():
        again = DecodePathSpec.from_dict(spec.to_dict())
        assert again == spec
        assert again.fingerprint == spec.fingerprint


def test_unknown_path_raises() -> None:
    with pytest.raises(KeyError, match="unknown decode path"):
        get_decode_path("nope")


def test_bad_completion_kind_raises() -> None:
    with pytest.raises(ValueError, match="completion_kind"):
        DecodePathSpec(
            path_id="x",
            description="d",
            generation_entry="g",
            completion_kind="wild",  # type: ignore[arg-type]
            grammar_policy="p",
            seed_policy="s",
            expected_fallback="f",
            impl_version="v",
        )


def test_codec_override_for_unsupported_codec_raises() -> None:
    with pytest.raises(ValueError, match="unsupported codecs"):
        DecodePathSpec(
            path_id="x",
            description="d",
            generation_entry="g",
            completion_kind="greedy",
            grammar_policy="p",
            seed_policy="s",
            expected_fallback="f",
            impl_version="v",
            supported_output_codecs=("choice",),
            codec_lever_overrides=(("lexer", (("compiler_decode_mode", "tree"),)),),
        )


def test_choice_checkpoint_supports_all_required_paths() -> None:
    results = compatible_decode_paths(
        model_family="twotower", output_codec="choice", output_contract_version=1
    )
    ok = {spec.path_id for spec, compatible, _ in results if compatible}
    assert set(REQUIRED_DECODE_PATH_IDS) <= ok


def test_non_twotower_only_supports_declared_control() -> None:
    results = compatible_decode_paths(
        model_family="grammar_diffusion", output_codec="compositional"
    )
    ok = {spec.path_id for spec, compatible, _ in results if compatible}
    assert ok == {"checkpoint_declared"}
    # Incompatible cells carry a stable non-coercion reason.
    reasons = [r for _s, compatible, r in results if not compatible]
    assert all(r for r in reasons)


def test_exact_or_compiler_preserves_representation_per_codec() -> None:
    spec = get_decode_path("current_exact_or_compiler")
    # Choice codec -> exact pushdown (no compiler-tree).
    choice = spec.resolve_config_overrides("choice")
    assert choice["compiler_decode_mode"] == "off"
    assert choice["allow_unconstrained_fallback"] is False
    assert choice["slot_contract_constrained_decode"] is True
    assert "schema_in_context" not in choice
    assert "slot_contract_in_context" not in choice
    # Surface/lexer -> compiler-tree greedy.
    lexer = spec.resolve_config_overrides("lexer")
    assert lexer["compiler_decode_mode"] == "tree"
    assert lexer["compiler_search_mode"] == "greedy"
    assert lexer["slot_contract_constrained_decode"] is True
    # A surface checkpoint is never coerced into a choice codec: the codec set
    # excludes nothing it supports, and an unknown codec is incompatible.
    ok, reason = spec.is_compatible(model_family="twotower", output_codec="mystery")
    assert ok is False and "coerce" in reason


def test_checkpoint_declared_applies_no_overrides() -> None:
    spec = get_decode_path("checkpoint_declared")
    assert spec.runtime_override_fields() == ()
    assert spec.resolve_config_overrides("choice") == {}
