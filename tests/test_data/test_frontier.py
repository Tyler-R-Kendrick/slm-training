"""Tests for the frozen frontier-artifact contract (F3 / SLM-4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.frontier import artifact_path, gold_content_hash
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data.catalog import classify_source_family
from slm_training.harnesses.train_data.synth import FrozenArtifactSynthesizer

HERO = 'root = Stack([hero], "column")\nhero_t = TextContent(":hero.title")\nhero = Card([hero_t])'
OTHER = 'root = Stack([a, b], "row")\na = Button(":x.y")\nb = Button(":z.w")'


def _gold(
    id: str = "train_hero_01",
    openui: str = HERO,
    prompt: str = "Create a hero card.",
    split: str = "train",
) -> ExampleRecord:
    return ExampleRecord(
        id=id,
        prompt=prompt,
        openui=openui,
        placeholders=[":hero.title"],
        split=split,
        source="fixture",
    )


def _write_artifact(
    root: Path,
    gold: ExampleRecord,
    *,
    skeleton: str | None = None,
    paraphrases: tuple[str, ...] = (),
    ladder: dict[str, str] | None = None,
) -> Path:
    gold_hash = gold_content_hash(gold.openui, gold.prompt)
    bundle = {
        "gold_id": gold.id,
        "gold_content_hash": gold_hash,
        "skeleton_openui": skeleton if skeleton is not None else gold.openui,
        "provenance": {"skill": "frontier-describe", "skill_version": "0.1.0"},
        "paraphrases": list(paraphrases),
        "ladder": dict(ladder or {}),
    }
    path = artifact_path(gold.id, gold_hash, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def test_gold_content_hash_is_deterministic_16_hex() -> None:
    value = gold_content_hash(HERO, "p")
    assert value == gold_content_hash(HERO, "p")
    assert len(value) == 16
    int(value, 16)
    assert gold_content_hash(HERO, "p") != gold_content_hash(OTHER, "p")


def test_synth_emits_paraphrase_and_ladder_rows(tmp_path: Path) -> None:
    gold = _gold()
    _write_artifact(
        tmp_path,
        gold,
        paraphrases=("Build a hero card.", "A card with a title."),
        ladder={"semantic": "A vertical card holding a heading.", "user": "I want a hero."},
    )
    rows = FrozenArtifactSynthesizer(root=tmp_path).expand(gold)
    assert len(rows) == 4
    assert all(r.openui == gold.openui for r in rows)  # skeleton is the target
    assert all(r.meta["parent_id"] == gold.id for r in rows)
    assert all(r.meta["task"] == "generation" for r in rows)
    assert {classify_source_family(r) for r in rows} == {
        "frontier_described",
        "frontier_semantic",
        "frontier_user",
    }
    ladder = [r for r in rows if r.meta.get("abstraction_level")]
    assert {r.meta["abstraction_level"] for r in ladder} == {"L1", "L4"}
    assert all("constraint_coverage" in r.meta for r in ladder)


def test_stale_gold_drops_artifact(tmp_path: Path) -> None:
    gold = _gold()
    _write_artifact(tmp_path, gold, paraphrases=("Build a hero card.",))
    # The gold changed → new content hash → the artifact filename no longer resolves.
    changed = ExampleRecord(
        id=gold.id, prompt=gold.prompt, openui=OTHER, placeholders=[":x.y"], split="train"
    )
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(changed) == []


@pytest.mark.parametrize("payload", [[], None, "invalid", "{"])
def test_malformed_artifact_drops_silently(tmp_path: Path, payload: object) -> None:
    gold = _gold()
    path = artifact_path(
        gold.id, gold_content_hash(gold.openui, gold.prompt), root=tmp_path
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    content = payload if isinstance(payload, str) and payload == "{" else json.dumps(payload)
    path.write_text(content, encoding="utf-8")
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(gold) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("gold_id", "other"), ("gold_content_hash", "0" * 16)],
)
def test_embedded_artifact_identity_must_match_path(
    tmp_path: Path, field: str, value: str
) -> None:
    gold = _gold()
    path = _write_artifact(tmp_path, gold, paraphrases=("Build a hero card.",))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(gold) == []


def test_held_out_artifact_is_never_described(tmp_path: Path) -> None:
    held = _gold(split="held_out")
    _write_artifact(tmp_path, held, paraphrases=("Build a hero card.",))
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(held) == []


def test_faithfulness_bind_rejects_mismatched_skeleton(tmp_path: Path) -> None:
    gold = _gold()
    # Filename hash matches the gold, but the described skeleton is a structurally
    # different program → reject (never let an unfaithful artifact through).
    _write_artifact(tmp_path, gold, skeleton=OTHER, paraphrases=("x",))
    assert FrozenArtifactSynthesizer(root=tmp_path).expand(gold) == []


def test_worklist_lists_train_golds_only(tmp_path: Path) -> None:
    from scripts.frontier_worklist import build_worklist

    records = tmp_path / "recs.jsonl"
    write_jsonl(
        records,
        [
            _gold(id="train_a", split="train"),
            ExampleRecord(
                id="held_b",
                prompt="p",
                openui=HERO,
                placeholders=[":hero.title"],
                split="held_out",
                source="fixture",
            ),
        ],
    )
    rows = build_worklist(records, frontier_root=tmp_path)
    assert [r["gold_id"] for r in rows] == ["train_a"]  # test/held golds excluded
    assert rows[0]["has_fresh_artifact"] is False
    _write_artifact(tmp_path, _gold(id="train_a"))
    rows2 = build_worklist(records, frontier_root=tmp_path)
    assert rows2[0]["has_fresh_artifact"] is True
