"""Cross-structure semantic dedup tests (lexical fallback engine)."""

from __future__ import annotations

import pytest

from slm_training.data.semantic_dedup import (
    apply_semantic_dedup,
    default_threshold,
    record_vectors,
    similarity_engine,
)
from slm_training.dsl.schema import ExampleRecord


def _record(record_id: str, prompt: str, openui: str) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        placeholders=[],
        split="train",
    )


_BASE_PROMPT = (
    "Design a profile summary card showing the member name, their role in the "
    "team, and a short biography paragraph underneath."
)
_BASE_OPENUI = (
    'root = Stack([card], "column")\n'
    'name = TextContent(":member.name")\n'
    'role = TextContent(":member.role")\n'
    'bio = TextContent(":member.bio")\n'
    "card = Card([name, role, bio])"
)


@pytest.fixture(autouse=True)
def _lexical_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLM_SEMANTIC_DEDUP_ENGINE", "lexical")


def test_paraphrase_with_different_structure_collapses() -> None:
    base = _record("a1_base", _BASE_PROMPT, _BASE_OPENUI)
    # Paraphrased prompt + a structurally different layout (extra wrapper),
    # invisible to structure-bucketed MinHash dedup.
    paraphrase = _record(
        "a2_paraphrase",
        _BASE_PROMPT.replace("short biography", "brief biography"),
        _BASE_OPENUI.replace("card = Card([name, role, bio])", "card = Card([name, bio, role])"),
    )
    distinct = _record(
        "b1_distinct",
        "Build a checkout form with quantity stepper and a pay button.",
        'root = Stack([qty, pay], "column")\n'
        'qty = Slider(":cart.qty")\n'
        'pay = Button(":cart.pay")',
    )
    kept, dropped = apply_semantic_dedup(
        [base, paraphrase, distinct], threshold=0.9
    )
    assert [record.id for record in kept] == ["a1_base", "b1_distinct"]
    assert len(dropped) == 1
    drop = dropped[0]
    assert drop["id"] == "a2_paraphrase"
    assert drop["duplicate_of"] == "a1_base"
    assert drop["reason"] == "semantic_cosine"
    assert drop["engine"] == "lexical-tfidf"
    assert drop["similarity"] >= 0.9


def test_deterministic_across_runs() -> None:
    records = [
        _record("r1", _BASE_PROMPT, _BASE_OPENUI),
        _record("r2", _BASE_PROMPT + " Keep it compact.", _BASE_OPENUI),
        _record(
            "r3",
            "A completely different dashboard of shipping statistics tiles.",
            'root = Stack([tiles], "row")\ntiles = TextContent(":stats.total")',
        ),
    ]
    first = apply_semantic_dedup(records, threshold=0.9)
    second = apply_semantic_dedup(records, threshold=0.9)
    assert [r.id for r in first[0]] == [r.id for r in second[0]]
    assert first[1] == second[1]


def test_zero_threshold_disables_the_pass() -> None:
    records = [
        _record("r1", _BASE_PROMPT, _BASE_OPENUI),
        _record("r2", _BASE_PROMPT, _BASE_OPENUI),
    ]
    kept, dropped = apply_semantic_dedup(records, threshold=0.0)
    assert len(kept) == 2 and not dropped


def test_lexical_engine_reported_with_default_threshold() -> None:
    assert similarity_engine() == "lexical-tfidf"
    assert default_threshold("lexical-tfidf") == 0.95
    assert default_threshold("embeddings") == 0.92
    vectors, engine = record_vectors([_record("v1", _BASE_PROMPT, _BASE_OPENUI)])
    assert engine == "lexical-tfidf"
    assert vectors.shape[0] == 1


def test_deliberate_variant_families_are_exempt() -> None:
    base = _record("a1_base", _BASE_PROMPT, _BASE_OPENUI)
    augmented = _record("a2_ns", _BASE_PROMPT, _BASE_OPENUI.replace(":member.", ":acme."))
    augmented.meta = {"source_family": "namespace_augment"}
    kept, dropped = apply_semantic_dedup([base, augmented], threshold=0.9)
    assert {record.id for record in kept} == {"a1_base", "a2_ns"}
    assert not dropped


def test_requested_embeddings_without_extra_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers installed; fail-closed path not reachable")
    except ImportError:
        pass
    monkeypatch.setenv("SLM_SEMANTIC_DEDUP_ENGINE", "embeddings")
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        similarity_engine()
