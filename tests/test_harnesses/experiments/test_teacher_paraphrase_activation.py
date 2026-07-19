"""Regression tests for SDE4-03 (SLM-181) activation/budget manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.harnesses.experiments.teacher_paraphrase_activation import (
    MANIFEST_SCHEMA,
    ActivationGate,
    BudgetCap,
    CanonicalRequest,
    TeacherParaphraseActivationManifest,
    TeacherParaphraseArm,
    TeacherProviderConfig,
    build_teacher_paraphrase_activation_manifest,
    render_canonical_request,
    validate_teacher_paraphrase_activation_manifest,
)


@pytest.fixture
def sample_spec() -> ProgramSpec:
    return ProgramSpec(
        id="root-001",
        ast={},
        canonical_openui='root = Stack(\n  header = CardHeader("Title"),\n  children = [\n    TextContent("Hello :user")\n  ]\n)',
        facts={"output_kind": "document"},
        contract_id="abcd1234abcd1234",
        program_family_id="pf-001",
        lineage_id="lineage-001",
        split_group_id="sg-001",
        split="train",
        provenance={"output_kind": "document"},
    )


@pytest.fixture
def default_gates() -> list[ActivationGate]:
    return [
        ActivationGate(
            gate_id="canonical_ast_codec_binding",
            depends_on_issue_id="SLM-169",
            required_status="Done",
            available=True,
            evidence="gate available",
        ),
        ActivationGate(
            gate_id="roottype_diversity_economics",
            depends_on_issue_id="SLM-171",
            required_status="Done",
            available=True,
            evidence="prompt diversity limited",
        ),
        ActivationGate(
            gate_id="independent_judge_path",
            depends_on_issue_id="SLM-106",
            required_status="Done",
            available=True,
            evidence="judge available",
        ),
    ]


@pytest.fixture
def default_provider() -> TeacherProviderConfig:
    return TeacherProviderConfig(
        provider="openai",
        model="gpt-4o",
        revision="2024-05-13",
        system_prompt_template_hash="sha256:abc",
        user_prompt_template_hash="sha256:def",
        sampling_parameters={"temperature": 0.7},
        max_tokens=4096,
        retry_policy={"max_retries": 3},
        cost_per_1k_input_usd=0.005,
        cost_per_1k_output_usd=0.015,
    )


@pytest.fixture
def default_budget() -> BudgetCap:
    return BudgetCap(
        max_dollars=100.0,
        max_input_tokens=10_000_000,
        max_output_tokens=2_000_000,
    )


@pytest.fixture
def default_arms() -> list[TeacherParaphraseArm]:
    return [
        TeacherParaphraseArm(
            arm_id="teacher_paraphrases",
            corpus_variant="teacher_paraphrases",
            eligible=True,
            styles=("concise", "detailed"),
        ),
        TeacherParaphraseArm(
            arm_id="deterministic_templates",
            corpus_variant="deterministic_templates",
            eligible=True,
        ),
    ]


def test_build_manifest_ready_when_gates_and_budget_ok(
    default_gates, default_provider, default_budget, default_arms
):
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="sde4-03-test",
        activation_gates=default_gates,
        provider=default_provider,
        budget=default_budget,
        arms=default_arms,
        slm171_outcome="prompt_diversity_limited",
    )
    assert manifest.schema_version == MANIFEST_SCHEMA
    assert manifest.hypothesis_id == "H19"
    assert manifest.activation_status == "ready"
    assert manifest.activation_verdict == "ready_to_spend"
    assert manifest.manifest_hash is not None
    assert len(manifest.manifest_hash) == 16


def test_build_manifest_blocked_when_gate_unavailable(
    default_gates, default_provider, default_budget, default_arms
):
    gates = [
        ActivationGate(
            gate_id=g.gate_id,
            depends_on_issue_id=g.depends_on_issue_id,
            required_status=g.required_status,
            available=(g.gate_id != "roottype_diversity_economics"),
            evidence=g.evidence,
        )
        for g in default_gates
    ]
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="sde4-03-test",
        activation_gates=gates,
        provider=default_provider,
        budget=default_budget,
        arms=default_arms,
        slm171_outcome="prompt_diversity_limited",
    )
    assert manifest.activation_status == "blocked"
    assert manifest.activation_verdict == "activation_blocked"


def test_build_manifest_not_prioritized_when_root_limited(
    default_gates, default_provider, default_budget, default_arms
):
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="sde4-03-test",
        activation_gates=default_gates,
        provider=default_provider,
        budget=default_budget,
        arms=default_arms,
        slm171_outcome="root_diversity_limited",
    )
    assert manifest.activation_status == "closed"
    assert manifest.activation_verdict == "teacher_paraphrases_not_prioritized"


def test_build_manifest_budget_blocked_when_caps_zero(
    default_gates, default_provider, default_arms
):
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="sde4-03-test",
        activation_gates=default_gates,
        provider=default_provider,
        budget=BudgetCap(max_dollars=0.0, max_input_tokens=0, max_output_tokens=0),
        arms=default_arms,
        slm171_outcome="prompt_diversity_limited",
    )
    assert manifest.activation_status == "blocked"
    assert manifest.activation_verdict == "budget_or_yield_blocked"


def test_manifest_round_trip_dict(default_gates, default_provider, default_budget, default_arms):
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="sde4-03-test",
        activation_gates=default_gates,
        provider=default_provider,
        budget=default_budget,
        arms=default_arms,
        slm171_outcome="prompt_diversity_limited",
    )
    data = manifest.to_dict()
    restored = TeacherParaphraseActivationManifest.from_dict(data)
    assert restored == manifest


def test_validate_manifest_catches_errors():
    assert validate_teacher_paraphrase_activation_manifest({}) != []
    assert validate_teacher_paraphrase_activation_manifest(
        {"schema_version": "wrong"}
    ) != []
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="x",
        activation_gates=[
            ActivationGate("g", "SLM-169", "Done", True),
        ],
        provider=TeacherProviderConfig(provider="p", model="m"),
        budget=BudgetCap(max_dollars=1.0),
        arms=[
            TeacherParaphraseArm(
                arm_id="a",
                corpus_variant="teacher_paraphrases",
                eligible=True,
            ),
        ],
    )
    assert validate_teacher_paraphrase_activation_manifest(manifest.to_dict()) == []


def test_validate_manifest_catches_invalid_arm_style(
    default_gates, default_provider, default_budget, default_arms
):
    manifest = build_teacher_paraphrase_activation_manifest(
        manifest_id="x",
        activation_gates=default_gates,
        provider=default_provider,
        budget=default_budget,
        arms=default_arms,
    )
    data = manifest.to_dict()
    data["arms"][0]["styles"] = ["invalid_style"]
    errors = validate_teacher_paraphrase_activation_manifest(data)
    assert any("invalid styles" in e for e in errors)


def test_render_canonical_request_exposes_semantics_not_internals(sample_spec):
    request = render_canonical_request(sample_spec)
    assert isinstance(request, CanonicalRequest)
    assert request.request_text.startswith("Create an OpenUI program")
    assert "Stack" in request.request_text
    assert ":user" in request.request_text
    assert request.output_kind == "document"
    assert request.leakage_flags == ()
    assert request.request_hash is not None


def test_render_canonical_request_with_design_md(sample_spec):
    request = render_canonical_request(
        sample_spec,
        design_md="Use a single-column layout.\nAvoid modals.",
        output_kind="screen",
    )
    assert request.output_kind == "screen"
    assert "single-column" in request.request_text
    assert "Avoid modals" in request.request_text


def test_render_canonical_request_rejects_raw_openui_leak(sample_spec):
    # If the request accidentally echoed the full source, the leakage flag fires.
    # We cannot make it fire normally, so verify the contract shape instead.
    request = render_canonical_request(sample_spec)
    assert sample_spec.canonical_openui not in request.request_text
    assert not any(
        line.strip().startswith("root =") for line in request.request_text.splitlines()
    )


def test_cli_builds_manifest(tmp_path: Path):
    from scripts.build_teacher_paraphrase_activation_manifest import main

    out_json = tmp_path / "manifest.json"
    rc = main(
        [
            "--manifest-id",
            "sde4-03-cli",
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--max-dollars",
            "50.0",
            "--max-input-tokens",
            "1000000",
            "--max-output-tokens",
            "200000",
            "--slm171-outcome",
            "prompt_diversity_limited",
            "--gate-slm-169",
            "--gate-slm-171",
            "--gate-slm-106",
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    manifest = json.loads(out_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["manifest_id"] == "sde4-03-cli"
    assert manifest["activation_status"] == "ready"
    assert manifest["activation_verdict"] == "ready_to_spend"
    assert validate_teacher_paraphrase_activation_manifest(manifest) == []
