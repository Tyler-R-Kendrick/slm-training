"""DSH2-04 (SLM-365): frame-fact paraphrases with leak detector + grounding floor."""

from __future__ import annotations

import hashlib

import pytest

from tests.casefiles import case_values

from slm_training.dsl.paraphrase_provenance import SamplingParamsV1
from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame
from slm_training.harnesses.train_data.frame_paraphrases import (
    NEAR_DUPLICATE_JACCARD,
    STYLE_PLAN,
    FrameLeakDetectorV1,
    FrameParaphraseFixtureProviderV1,
    FrameParaphraseRowV1,
    _Candidate,
    compute_diversity,
    dedupe_paraphrase_rows,
    enforce_grounding_floor,
    generate_answer_family_paraphrases,
    generate_frame_paraphrases,
)

CARD_SRC = (
    'a = TextContent(":ns.body", 14)\n'
    'root = Card([a], "filled", "column", 8, "start", "start", "nowrap")\n'
)
# Same canonical AST, different surface formatting -> equivalent frame.
CARD_SRC_RESPACED = (
    'a = TextContent( ":ns.body", 14 )\n'
    'root = Card( [a], "filled", "column", 8, "start", "start", "nowrap" )\n'
)
CALLOUT_SRC = 'root = Callout("info", ":ns.title", ":ns.desc", true)'

NOW = lambda: "2026-07-25T00:00:00+00:00"  # noqa: E731


@pytest.fixture(scope="module")
def schema() -> CAP1SchemaV1:
    return CAP1SchemaV1.from_pack("openui")


@pytest.fixture()
def frame(schema: CAP1SchemaV1):
    return derive_semantic_frame(CARD_SRC, schema)


@pytest.fixture()
def paraphrase_set(frame):
    return generate_frame_paraphrases(
        frame, provider=FrameParaphraseFixtureProviderV1(), now=NOW
    )


class _StyledProvider:
    """Wraps the fixture provider and rewrites one style's output."""

    provider_id = "styled-fixture"
    model_id = "styled-renderer"
    model_revision = "v1"

    def __init__(self, style: str, replacement) -> None:
        self._inner = FrameParaphraseFixtureProviderV1()
        self._style = style
        self._replacement = replacement

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, *, prompt_text: str, **kwargs) -> str:
        text = self._inner.generate(prompt_text=prompt_text, **kwargs)
        if (
            f'"style": "{self._style}"' in prompt_text
            or f'"style":"{self._style}"' in prompt_text
        ):
            return self._replacement(text)
        return text


def _required_count(frame) -> int:
    return len(frame.facts_by_category("required"))


# --------------------------------------------------------------------------
# Style plan generation
# --------------------------------------------------------------------------


def test_style_plan_is_plan_declared() -> None:
    assert STYLE_PLAN == ("concise", "descriptive", "imperative", "compositional")


def test_all_four_styles_generate_from_frame(paraphrase_set) -> None:
    assert [row.style for row in paraphrase_set.rows] == list(STYLE_PLAN)
    assert paraphrase_set.rejected_rows == ()
    for style in STYLE_PLAN:
        assert paraphrase_set.per_style_counts[style] == {"accepted": 1, "rejected": 0}


def test_every_row_covers_all_required_facts(frame, paraphrase_set) -> None:
    for row in paraphrase_set.rows:
        assert len(row.required_facts_covered) == _required_count(frame)
        assert row.forbidden_facts_absent is True
        assert row.leak_check_pass is True
        assert (
            row.prompt_sha256
            == hashlib.sha256(row.prompt_text.encode("utf-8")).hexdigest()
        )


def test_callout_frame_generates_all_styles(schema) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    result = generate_frame_paraphrases(
        frame, provider=FrameParaphraseFixtureProviderV1(), now=NOW
    )
    assert [row.style for row in result.rows] == list(STYLE_PLAN)


# --------------------------------------------------------------------------
# Required coverage / forbidden absence / grounding floor
# --------------------------------------------------------------------------


def test_style_missing_a_required_fact_is_rejected(frame) -> None:
    provider = _StyledProvider(
        "descriptive", lambda text: text.replace("the variant is filled", "styled")
    )
    result = generate_frame_paraphrases(frame, provider=provider, now=NOW)
    styles = [row.style for row in result.rows]
    assert "descriptive" not in styles
    assert sorted(styles) == sorted(s for s in STYLE_PLAN if s != "descriptive")
    rejected = [r for r in result.rejected_rows if r.style == "descriptive"]
    assert len(rejected) == 1
    assert rejected[0].reason == "missing_required_facts"
    assert "the variant is filled" in rejected[0].detail
    assert result.rejection_counts["missing_required_facts"] == 1


def test_forbidden_fact_assertion_is_rejected(schema) -> None:
    frame = derive_semantic_frame(CALLOUT_SRC, schema)
    forbidden = frame.facts_by_category("forbidden")
    assert forbidden, "callout fixture must have closed-domain forbidden facts"
    provider = _StyledProvider(
        "imperative", lambda text: text + " Also, the visible is false."
    )
    result = generate_frame_paraphrases(frame, provider=provider, now=NOW)
    imperative = [r for r in result.rejected_rows if r.style == "imperative"]
    assert len(imperative) == 1
    assert imperative[0].reason == "forbidden_fact_present"


def test_enforce_grounding_floor_unit(frame) -> None:
    detector = FrameLeakDetectorV1.from_frame(frame)
    good = _Candidate(
        style="concise",
        text="a card component; the variant is filled",
        provenance={},
        coverage_keys=("a card component", "the variant is filled"),
        forbidden_phrases=("the variant is outlined",),
        leak_detector=detector,
    )
    leaking = _Candidate(
        style="imperative",
        text="build a card component with Card( and the variant is filled",
        provenance={},
        coverage_keys=("a card component", "the variant is filled"),
        forbidden_phrases=(),
        leak_detector=detector,
    )
    accepted, rejected = enforce_grounding_floor([good, leaking])
    assert [r.style for r in accepted] == ["concise"]
    assert [r.style for r in rejected] == ["imperative"]
    assert rejected[0].reason == "leak_detected"


# --------------------------------------------------------------------------
# Leak detector
# --------------------------------------------------------------------------


def test_detector_built_from_frame(frame) -> None:
    detector = FrameLeakDetectorV1.from_frame(frame)
    assert "Card" in detector.component_names
    assert "TextContent" in detector.component_names
    assert ":ns.body" in detector.marker_surfaces
    assert len(detector.answer_digests) == 2


@pytest.mark.parametrize(
    "text,fragment",
    case_values(__file__, "test_detector_catches_leaks"),
)
def test_detector_catches_leaks(frame, text, fragment) -> None:
    ok, reasons = FrameLeakDetectorV1.from_frame(frame).check(text)
    assert not ok
    assert fragment in reasons


def test_detector_catches_answer_digest(frame) -> None:
    detector = FrameLeakDetectorV1.from_frame(frame)
    digest = hashlib.sha256(frame.canonical_source.encode("utf-8")).hexdigest()
    ok, reasons = detector.check(f"answer is {digest}")
    assert not ok
    assert "answer_digest" in reasons


def test_stop_rule_drops_leaking_style_with_reason(frame) -> None:
    provider = _StyledProvider("imperative", lambda text: text + " via Card(.")
    result = generate_frame_paraphrases(frame, provider=provider, now=NOW)
    styles = [row.style for row in result.rows]
    assert "imperative" not in styles
    assert sorted(styles) == ["compositional", "concise", "descriptive"]
    dropped = [r for r in result.rejected_rows if r.style == "imperative"]
    assert len(dropped) == 1
    assert dropped[0].reason == "leak_detected"
    assert "dsl_syntax:Card(" in dropped[0].detail
    assert result.rejection_counts["leak_detected"] == 1
    assert result.per_style_counts["imperative"] == {"accepted": 0, "rejected": 1}


def test_targets_stay_symbol_only(frame, paraphrase_set) -> None:
    detector = FrameLeakDetectorV1.from_frame(frame)
    for row in paraphrase_set.rows:
        assert frame.canonical_source not in row.prompt_text
        ok, _ = detector.check(row.prompt_text)
        assert ok
        # Serialized provenance carries digests, never the answer surface.
        assert frame.canonical_source not in str(row.style_provider_provenance)


# --------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------


def test_diversity_metrics_per_frame_and_style(paraphrase_set) -> None:
    diversity = paraphrase_set.diversity
    assert [s.style for s in diversity.per_style] == list(STYLE_PLAN)
    for stat in diversity.per_style:
        assert stat.token_count > 0
        assert 0.0 < stat.distinct_token_ratio <= 1.0
    expected_pairs = {
        "concise|descriptive",
        "concise|imperative",
        "concise|compositional",
        "descriptive|imperative",
        "descriptive|compositional",
        "imperative|compositional",
    }
    assert set(diversity.pairwise_token_jaccard) == expected_pairs
    assert diversity.lexical_diversity > 0.0
    assert diversity.structural_diversity == 1.0


def test_compute_diversity_empty() -> None:
    diversity = compute_diversity(())
    assert diversity.per_style == ()
    assert diversity.lexical_diversity == 0.0
    assert diversity.structural_diversity == 0.0


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------


def _row(style: str, text: str, signature: str | None = None) -> FrameParaphraseRowV1:
    return FrameParaphraseRowV1(
        style=style,
        prompt_text=text,
        prompt_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        required_facts_covered=("a card component",),
        forbidden_facts_absent=True,
        leak_check_pass=True,
        style_provider_provenance={},
        template_signature=signature or f"sig-{style}-{text[:8]}",
    )


def test_dedup_exact_duplicate() -> None:
    kept, rejected = dedupe_paraphrase_rows(
        [
            _row("concise", "layout one two three", signature="s1"),
            _row("concise", "layout one two three", signature="s2"),
        ]
    )
    assert len(kept) == 1
    assert [r.reason for r in rejected] == ["exact_duplicate"]


def test_dedup_near_duplicate_same_style() -> None:
    base = "layout alpha beta gamma delta epsilon zeta eta theta"
    near = "layout alpha beta gamma delta epsilon zeta eta theta extra"
    kept, rejected = dedupe_paraphrase_rows(
        [_row("concise", base, "s1"), _row("concise", near, "s2")],
        near_jaccard=NEAR_DUPLICATE_JACCARD,
    )
    assert len(kept) == 1
    assert [r.reason for r in rejected] == ["near_duplicate"]


def test_dedup_repeated_style_template() -> None:
    kept, rejected = dedupe_paraphrase_rows(
        [
            _row("concise", "totally different words here", signature="same"),
            _row("concise", "nothing alike at all really", signature="same"),
        ]
    )
    assert len(kept) == 1
    assert [r.reason for r in rejected] == ["repeated_style_template"]


def test_dedup_never_collapses_distinct_styles() -> None:
    # Identical text across styles is still an exact duplicate.
    text = "layout alpha beta gamma delta"
    kept, rejected = dedupe_paraphrase_rows(
        [_row("concise", text, "s1"), _row("descriptive", text, "s2")]
    )
    assert len(kept) == 1
    assert [r.reason for r in rejected] == ["exact_duplicate"]
    # Near-identical text across styles is NOT deduplicated: near/template
    # rules are same-style only, so distinct styles of one frame survive.
    kept, rejected = dedupe_paraphrase_rows(
        [
            _row("concise", "layout alpha beta gamma delta", "s1"),
            _row("descriptive", "layout alpha beta gamma delta extra", "s2"),
        ]
    )
    assert len(kept) == 2
    assert rejected == ()


def test_cross_family_dedup_links_one_answer(schema) -> None:
    frame_a = derive_semantic_frame(CARD_SRC, schema)
    frame_b = derive_semantic_frame(CARD_SRC_RESPACED, schema)
    assert frame_a.equivalent_to(frame_b)
    family = generate_answer_family_paraphrases(
        [frame_a, frame_b], provider=FrameParaphraseFixtureProviderV1(), now=NOW
    )
    assert len(family.frame_fingerprints) == 2
    # Equivalent frames render identical prompts -> every second-frame row dedups.
    assert len(family.rows) == len(STYLE_PLAN)
    assert family.rejection_counts.get("exact_duplicate", 0) == len(STYLE_PLAN)
    expected = hashlib.sha256(frame_a.canonical_source.encode("utf-8")).hexdigest()
    assert family.canonical_answer_fingerprint == expected
    assert set(family.equivalence_fingerprints) == {expected}


# --------------------------------------------------------------------------
# Linking + provenance + determinism
# --------------------------------------------------------------------------


def test_accepted_rows_link_canonical_answer(frame, paraphrase_set) -> None:
    expected = hashlib.sha256(frame.canonical_source.encode("utf-8")).hexdigest()
    assert paraphrase_set.canonical_answer_fingerprint == expected
    assert paraphrase_set.equivalence_fingerprints == ()


def test_equivalence_fingerprints_recorded(frame) -> None:
    result = generate_frame_paraphrases(
        frame,
        provider=FrameParaphraseFixtureProviderV1(),
        equivalence_fingerprints=("eq-a", "eq-b"),
        now=NOW,
    )
    assert result.equivalence_fingerprints == ("eq-a", "eq-b")


def test_row_provenance_uses_dsh2_03_contract(paraphrase_set) -> None:
    for row in paraphrase_set.rows:
        prov = row.style_provider_provenance
        assert prov["provider_id"] == "slm365-frame-fixture"
        assert prov["model_id"] == "frame-style-renderer"
        assert prov["model_revision"] == "v1"
        request = prov["request"]
        assert request["prompt_digest"]
        assert request["system_digest"]
        assert request["schema_fingerprint"]
        assert request["frame_fingerprint"]
        assert (
            prov["response"]["response_digest"]
            == hashlib.sha256(row.prompt_text.encode("utf-8")).hexdigest()
        )


def test_determinism_via_offline_provider(frame) -> None:
    kwargs = dict(
        provider=FrameParaphraseFixtureProviderV1(),
        sampling=SamplingParamsV1(),
        seed=7,
        now=NOW,
    )
    first = generate_frame_paraphrases(frame, **kwargs)
    second = generate_frame_paraphrases(frame, **kwargs)
    assert first.to_dict() == second.to_dict()


def test_serialization_roundtrip_shape(paraphrase_set) -> None:
    payload = paraphrase_set.to_dict()
    assert payload["schema_version"] == "frame_paraphrases/v1"
    assert set(payload["rows"][0]) == {
        "style",
        "prompt_text",
        "prompt_sha256",
        "required_facts_covered",
        "forbidden_facts_absent",
        "leak_check_pass",
        "style_provider_provenance",
        "template_signature",
    }
