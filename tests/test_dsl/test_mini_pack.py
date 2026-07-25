"""DSH2-07 (SLM-367): mini-flow pack completeness + no-OpenUI generic-code audit."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.dsl import mini_pack as mp
from slm_training.dsl.semantic_frame import verify_single_fact_change

MINI_SRC = 'let owner = "ops"\ntask build { step ":f.desc" mode fast retry 2 }\n'


@pytest.fixture(scope="module")
def bundle() -> mp.MiniPackBundleV1:
    return mp.build_mini_pack()


# --------------------------------------------------------------------------
# Completeness conformance (DSH1-01-style, fail-closed)
# --------------------------------------------------------------------------


def test_complete_pack_passes_conformance(bundle) -> None:
    report = mp.require_complete_for_generic_claim(bundle)
    assert report.complete is True
    assert report.missing == ()
    assert dict(report.probe_failures) == {}


def test_declared_requirements() -> None:
    assert set(mp.GENERIC_CLAIM_REQUIREMENTS) == {
        "grammar_backend",
        "canonicalizer",
        "static_validator",
        "schema_contract",
        "semantic_frame_provider",
        "marker_policy",
        "fragment_parse",
        "grammar_capability_authority",
    }


def test_missing_authority_fails_closed(bundle) -> None:
    incomplete = replace(bundle, pack=replace(bundle.pack, grammar_capability_authority=None))
    report = mp.completeness_report(incomplete)
    assert report.complete is False
    assert "grammar_capability_authority" in report.missing
    with pytest.raises(mp.PackIncompleteError):
        mp.require_complete_for_generic_claim(incomplete)


def test_missing_frame_provider_fails_closed(bundle) -> None:
    incomplete = replace(bundle, derive_frame=None)  # type: ignore[arg-type]
    report = mp.completeness_report(incomplete)
    assert report.complete is False
    assert "semantic_frame_provider" in report.missing
    with pytest.raises(mp.PackIncompleteError):
        mp.require_complete_for_generic_claim(incomplete)


def test_missing_schema_contract_fails_closed(bundle) -> None:
    incomplete = replace(bundle, schema=None)  # type: ignore[arg-type]
    report = mp.completeness_report(incomplete)
    assert report.complete is False
    assert "schema_contract" in report.missing


# --------------------------------------------------------------------------
# Generic-code audit: no OpenUI import anywhere in the mini pack path
# --------------------------------------------------------------------------


def test_mini_pack_imports_no_openui() -> None:
    source = Path(mp.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(alias.name for alias in node.names if "openui" in alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and "openui" in node.module:
            offenders.append(node.module)
    assert offenders == []


def test_mini_grammar_has_no_openui_surface() -> None:
    lines = [
        line
        for line in mp.GRAMMAR_PATH.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("//")
    ]
    grammar = "\n".join(lines).lower()
    assert "openui" not in grammar
    assert "stack" not in grammar and "textcontent" not in grammar


def test_mini_frame_derivation_needs_no_openui_pack(bundle, monkeypatch) -> None:
    # If mini frame derivation touched the OpenUI pack, forcing the builtin
    # pack registry to explode would break it.
    monkeypatch.setenv("SLM_GRAMMAR_DSL", "mini-flow")
    frame = bundle.derive_frame(MINI_SRC)
    assert frame.dsl == mp.DSL_ID
    assert frame.canonical_source == mp.mini_canonicalize(MINI_SRC)


# --------------------------------------------------------------------------
# Capability behavior
# --------------------------------------------------------------------------


def test_canonicalize_idempotent_and_normalizes() -> None:
    messy = 'task   build {  mode fast    step ":f.desc" }\n\n'
    canon = mp.mini_canonicalize(messy)
    assert canon == 'task build { mode fast step ":f.desc" }\n'
    assert mp.mini_canonicalize(canon) == canon


def test_static_validator_rejects_garbage() -> None:
    with pytest.raises(Exception):
        mp.mini_static_validate("task { mode fast }")
    with pytest.raises(Exception):
        mp.mini_static_validate("task x { mode slow }")  # outside closed domain
    mp.mini_static_validate(MINI_SRC)


def test_frame_facts_and_closed_domain(bundle) -> None:
    frame = bundle.derive_frame(MINI_SRC)
    required = frame.facts_by_category("required")
    closed = [f for f in required if f.predicate == "closed_value"]
    assert [(f.schema_field, f.value) for f in closed] == [("mode", "fast")]
    forbidden = frame.facts_by_category("forbidden")
    assert [(f.schema_field, f.value) for f in forbidden] == [("mode", "safe")]
    placeholders = [f for f in required if f.predicate == "content_placeholder_id"]
    assert [f.value for f in placeholders] == [":f.desc"]
    bindings = [f for f in required if f.predicate == "let_binding"]
    assert [(f.schema_field, f.value) for f in bindings] == [("owner", "ops")]


def test_single_fact_counterfactual_proof(bundle) -> None:
    before = bundle.derive_frame(MINI_SRC)
    after = bundle.derive_frame(MINI_SRC.replace("mode fast", "mode safe"))
    proof = verify_single_fact_change(before, after)
    assert proof.single_fact_change is True
    assert proof.schema_field == "mode"
    assert (proof.old_value, proof.new_value) == ("fast", "safe")


def test_fragment_parse(bundle) -> None:
    assert bundle.pack.require("fragment_parser")("mode safe", "fragment", "prop_stmt") == "mode safe"
    with pytest.raises(Exception):
        bundle.pack.require("fragment_parser")("mode slow", "fragment", "prop_stmt")


def test_oracle_honesty_label(bundle) -> None:
    verdict = bundle.pack.oracle(MINI_SRC)
    assert verdict["valid"] is True
    assert verdict["reward_label"] == "syntax_only"
    assert bundle.pack.oracle("task { mode fast }")["valid"] is False


def test_marker_policy(bundle) -> None:
    contract = bundle.pack.placeholder_policy.slot_contract(MINI_SRC)
    assert contract == (":f.desc",)
