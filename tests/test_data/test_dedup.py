"""P1a fuzzy + semantic dedup tests."""

from __future__ import annotations

from slm_training.data.dedup import (
    apply_fuzzy_dedup,
    apply_semantic_cluster_cap,
    binding_pattern_cluster,
    cluster_exposure_stats,
    family_priority,
    jaccard_from_signatures,
    minhash_signature,
    prompt_semantic_cluster,
)
from slm_training.dsl.schema import ExampleRecord


def _rec(rid: str, prompt: str, openui: str, family: str = "rico_real") -> ExampleRecord:
    return ExampleRecord(
        id=rid,
        prompt=prompt,
        openui=openui,
        split="train",
        source=family,
        meta={"source_family": family, "root_parent_id": rid},
    )


HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    "hero = Card([hero_title])"
)


def test_minhash_identical_texts_jaccard_one() -> None:
    a = minhash_signature("hello world layout")
    b = minhash_signature("hello world layout")
    assert a == b
    assert jaccard_from_signatures(a, b) == 1.0


def test_fuzzy_dedup_collapses_near_duplicates_preferring_priority() -> None:
    # Identical payload → Jaccard 1.0; human_feedback wins over paraphrase.
    low = _rec("a", "Hero card layout", HERO, family="prompt_paraphrase")
    high = _rec("b", "Hero card layout", HERO, family="human_feedback")
    kept, dropped = apply_fuzzy_dedup([low, high], threshold=0.92)
    assert len(kept) == 1
    assert kept[0].id == "b"
    assert dropped and dropped[0]["reason"] == "fuzzy_minhash"


def test_semantic_cluster_cap_keeps_max() -> None:
    records = [
        _rec(f"r{i}", "Hero card layout", HERO, family="rico_real") for i in range(5)
    ]
    kept, dropped = apply_semantic_cluster_cap(records, max_per_cluster=2)
    assert len(kept) == 2
    assert len(dropped) == 3
    assert family_priority("human_feedback") > family_priority("rico_real")


def test_prompt_and_binding_clusters_stable() -> None:
    a = prompt_semantic_cluster("hero card layout")
    b = prompt_semantic_cluster("hero card layout")
    assert a == b
    assert binding_pattern_cluster(HERO) == binding_pattern_cluster(HERO)


def test_cluster_exposure_stats() -> None:
    records = [_rec("a", "Hero", HERO), _rec("b", "Hero", HERO), _rec("c", "CTA", HERO)]
    stats = cluster_exposure_stats(records)
    assert stats["unique_clusters"] >= 1
    assert "p50" in stats["records_per_cluster"]
