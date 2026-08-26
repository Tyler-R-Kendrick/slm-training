"""Capability-aware SemanticFrameV1 prompt rendering."""

from __future__ import annotations

import re

from slm_training.dsl.parser import validate
from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame
from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    PromptSurface,
    PromptSurfacePolicy,
    RendererFamily,
    tiny_corpus_generation_policy,
)
from slm_training.harnesses.train_data.semantic_prompts import (
    CUE_INTERVENTIONS,
    PROVIDER_ID,
    SCHEMA_VERSION,
    apply_cue_intervention,
    render_prompt_candidates,
)

SRC = (
    'root = Stack([title, cta], "column")\n'
    'title = TextContent(":slot_0")\n'
    'cta = Button(":slot_1")\n'
)
_GENERIC = (
    "ui",
    "screen",
    "layout",
    "page",
    "interface",
    "build",
    "create",
    "generate",
    "design",
)
_PROVENANCE_KEYS = (
    "ast_path",
    "schema_field",
    "canonical_source",
    "schema_fingerprint",
    "provider_id",
    "renderer_family",
    "invariance_group_id",
)


def _frame(source: str = SRC):
    return derive_semantic_frame(validate(source), CAP1SchemaV1.from_pack("openui"))


def _nl_policy(*, prompts_per_root: int = 4) -> PromptSurfacePolicy:
    base = tiny_corpus_generation_policy(capability=Capability.CAP1_SEMANTICS).prompts
    return PromptSurfacePolicy(
        surface=base.surface,
        prompts_per_root=prompts_per_root,
        renderer_families=base.renderer_families,
        hard_forbidden_cues=base.hard_forbidden_cues,
        balanced_generic_cues=base.balanced_generic_cues,
    )


def _render(
    *, source: str = SRC, policy: PromptSurfacePolicy | None = None, seed: int = 0
):
    return render_prompt_candidates(
        _frame(source),
        policy or _nl_policy(),
        root_identity="root-fixture",
        seed=seed,
    )


def test_schema_and_provider_are_offline() -> None:
    result = _render()
    assert SCHEMA_VERSION == "semantic_prompts/v1"
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.provider_id == PROVIDER_ID == "offline_fixture_renderer"
        assert candidate.provenance["root_identity"] == "root-fixture"
        assert candidate.required_fact_ids
        asserted = candidate.provenance["required_fact_phrases"]
        target = candidate.provenance["target_fact_phrases"]
        assert set(asserted) <= set(target)
        assert all(
            phrase.casefold() in candidate.prompt_text.casefold() for phrase in asserted
        )
        assert "generation_policy" not in candidate.provenance
        assert candidate.sha == candidate.sha
        payload = candidate.to_dict()
        assert set(payload) == {
            "prompt_text",
            "renderer_family",
            "renderer_version",
            "surface",
            "required_fact_ids",
            "house_style_fact_ids",
            "forbidden_fact_ids",
            "cue_intervention",
            "invariance_group_id",
            "provider_id",
            "provenance",
        }


def test_simplified_nl_contains_zero_openui() -> None:
    for candidate in _render().candidates:
        assert "openui" not in candidate.prompt_text.lower()


def test_simplified_nl_has_no_dsl_binders_markers_hashes_or_ids() -> None:
    families = {item.value for item in RendererFamily}
    for candidate in _render().candidates:
        text = candidate.prompt_text
        assert "Stack(" not in text
        assert "TextContent(" not in text
        assert "Button(" not in text
        assert "root =" not in text
        assert "title =" not in text
        assert "cta =" not in text
        assert not re.search(r"\bcta\b", text, flags=re.IGNORECASE)
        assert ":slot_0" not in text
        assert ":slot_1" not in text
        assert "$" not in text
        assert not re.search(r"\b[0-9a-f]{32,}\b", text, flags=re.IGNORECASE)
        for family_id in families:
            assert family_id not in text
        for key in _PROVENANCE_KEYS:
            assert key not in text
        assert candidate.provenance["root_identity"] not in text


def test_simplified_prompt_does_not_list_complete_inventory() -> None:
    types = ("stack", "textcontent", "text content", "button")
    for candidate in _render().candidates:
        lowered = candidate.prompt_text.lower()
        mentioned = {
            name for name in types if re.search(rf"\b{re.escape(name)}\b", lowered)
        }
        assert not {"stack", "button"} <= mentioned or (
            "textcontent" not in mentioned and "text content" not in mentioned
        )
        assert not (
            re.search(r"\bstack\b", lowered)
            and ("text content" in lowered or "textcontent" in lowered)
            and re.search(r"\bbutton\b", lowered)
        )


def test_ordinary_user_terms_remain_allowed() -> None:
    joined = " ".join(item.prompt_text for item in _render().candidates).lower()
    assert "button" in joined


def test_one_prompt_omits_all_generic_cues() -> None:
    result = _render(policy=_nl_policy(prompts_per_root=2))
    assert result.candidates
    assert any(
        not any(
            re.search(rf"\b{re.escape(cue)}\b", item.prompt_text, flags=re.IGNORECASE)
            for cue in _GENERIC
        )
        for item in result.candidates
    )


def test_renderer_families_do_not_share_one_fixed_prefix() -> None:
    texts = [item.prompt_text for item in _render().candidates]
    assert len(texts) >= 2
    prefixes = [text.split()[0].lower() for text in texts]
    assert len(set(prefixes)) > 1
    assert not all(
        text.split()[0].lower() in {"create", "build", "design", "generate"}
        for text in texts
    )
    assert not any(text.lower().startswith("create an ui") for text in texts)


def test_cue_interventions_preserve_target_identity() -> None:
    frame = _frame()
    candidate = _render().candidates[0]
    for intervention in CUE_INTERVENTIONS:
        edited = apply_cue_intervention(candidate, intervention, frame=frame)
        assert edited.invariance_group_id == candidate.invariance_group_id
        assert edited.required_fact_ids == candidate.required_fact_ids
        assert edited.forbidden_fact_ids == candidate.forbidden_fact_ids
        assert edited.surface == candidate.surface
        assert edited.cue_intervention == intervention
        assert "openui" not in edited.prompt_text.lower()


def test_grammar_schema_may_mention_technical_terms() -> None:
    grammar = tiny_corpus_generation_policy(capability=Capability.CAP0_GRAMMAR).prompts
    technical = render_prompt_candidates(
        _frame(), grammar, root_identity="root-fixture", seed=0
    )
    assert technical.candidates
    assert all(
        item.provenance["required_fact_phrases"] for item in technical.candidates
    )
    assert any(
        "openui" in item.prompt_text.lower() or "Stack" in item.prompt_text
        for item in technical.candidates
    )
    simplified = _render()
    assert simplified.candidates
    assert all(
        "openui" not in item.prompt_text.lower() for item in simplified.candidates
    )
    assert all("dsl" not in item.prompt_text.lower() for item in simplified.candidates)


def test_deterministic_same_frame_policy_seed_root() -> None:
    frame = _frame()
    policy = _nl_policy()
    first = render_prompt_candidates(
        frame, policy, root_identity="root-fixture", seed=3
    )
    second = render_prompt_candidates(
        frame, policy, root_identity="root-fixture", seed=3
    )
    assert first.to_dict() == second.to_dict()
    assert [item.sha for item in first.candidates] == [
        item.sha for item in second.candidates
    ]


def test_row_stack_is_user_intent_not_dsl() -> None:
    source = (
        'root = Stack([title, cta], "row")\n'
        'title = TextContent(":slot_0")\n'
        'cta = Button(":slot_1")\n'
    )
    texts = [item.prompt_text.lower() for item in _render(source=source).candidates]
    assert any("side by side" in text or "beside" in text for text in texts)
    assert all("row stack" not in text for text in texts)
    assert all("use a row" not in text for text in texts)


def test_transformation_surface_is_edit_intent() -> None:
    policy = tiny_corpus_generation_policy(capability=Capability.CAP2_TRANSFORM).prompts
    result = render_prompt_candidates(
        _frame(), policy, root_identity="root-fixture", seed=0
    )
    assert result.candidates
    assert all(item.renderer_family == "transformation" for item in result.candidates)
    assert all(
        item.prompt_text.lower().startswith("change")
        or "change" in item.prompt_text.lower()
        for item in result.candidates
    )
    assert all(
        item.surface == PromptSurface.TRANSFORMATION_NL.value
        for item in result.candidates
    )
