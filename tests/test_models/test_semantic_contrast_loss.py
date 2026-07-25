"""Tests for the SLM-292 (AP-010) default-off semantic-contrast objective.

Pure loss/sampling/loader tests -- no TwoTowerModel construction here (that
lives in ``tests/test_harnesses/model_build/test_twotower.py`` alongside the
other optional-loss bit-exact-when-disabled tests). ``rep_fn`` is a small
deterministic stand-in for the real context encoder + pooling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slm_training.models.semantic_contrast_loss import (
    ContrastPairSample,
    SemanticContrastStepResult,
    compute_semantic_contrast_step,
    load_contrast_pairs,
    pairwise_margin_contrast_loss,
    sample_contrast_pairs,
)

REAL_CORPUS = Path(
    "src/slm_training/resources/data/eval/openui_hard_valid_v1/pairs.jsonl"
)


def _fixture_pair(
    pair_id: str,
    *,
    family: str = "content",
    transform_id: str = "content_swap_family",
    admitted: bool = True,
    prompt: str = "Generate a button and text",
    positive_openui: str = 'root = Stack([b1], "column")\nb1 = Button(":b1.label")',
    negative_openui: str = 'root = Stack([b1], "column")\nb1 = TextContent(":b1.label")',
    split: str = "train",
) -> dict:
    return {
        "pair_id": pair_id,
        "source_program_id": f"src_{pair_id}",
        "family": family,
        "transform_id": transform_id,
        "admitted": admitted,
        "admission_reason": "gate_pass" if admitted else "negative_passed",
        "positive": {
            "severity": "moderate",
            "record": {
                "id": f"{pair_id}_pos",
                "prompt": prompt,
                "openui": positive_openui,
                "placeholders": [":b1.label"],
                "split": split,
                "source": "fixture",
                "meta": {},
            },
        },
        "negative": {
            "severity": "moderate",
            "record": {
                "id": f"{pair_id}_neg",
                "prompt": prompt,
                "openui": negative_openui,
                "placeholders": [":b1.label"],
                "split": split,
                "source": "fixture",
                "meta": {},
            },
        },
    }


def _write_fixture_corpus(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _stub_rep_fn(texts: list[str]) -> torch.Tensor:
    """Deterministic bag-of-characters embedding -- no encoder required."""
    d = 16
    out = torch.zeros(len(texts), d)
    for i, text in enumerate(texts):
        for j, ch in enumerate(text):
            out[i, j % d] += float(ord(ch) % 7)
    return out


# ---------------------------------------------------------------------------
# load_contrast_pairs
# ---------------------------------------------------------------------------


def test_load_contrast_pairs_filters_non_admitted(tmp_path: Path) -> None:
    rows = [
        _fixture_pair("p1", admitted=True),
        _fixture_pair("p2", admitted=False),
        _fixture_pair("p3", admitted=True, family="binding", transform_id="binding_swap_symbol"),
    ]
    corpus = _write_fixture_corpus(tmp_path / "pairs.jsonl", rows)
    pairs = load_contrast_pairs(corpus)
    assert [p.pair_id for p in pairs] == ["p1", "p3"]
    assert all(isinstance(p, ContrastPairSample) for p in pairs)


def test_load_contrast_pairs_family_and_split_filters(tmp_path: Path) -> None:
    rows = [
        _fixture_pair("p1", family="content"),
        _fixture_pair("p2", family="binding", transform_id="binding_swap_symbol"),
        _fixture_pair("p3", family="content", split="held_out"),
    ]
    corpus = _write_fixture_corpus(tmp_path / "pairs.jsonl", rows)
    only_content_train = load_contrast_pairs(corpus, families=["content"], split="train")
    assert [p.pair_id for p in only_content_train] == ["p1"]


def test_load_contrast_pairs_respects_limit(tmp_path: Path) -> None:
    rows = [_fixture_pair(f"p{i}") for i in range(10)]
    corpus = _write_fixture_corpus(tmp_path / "pairs.jsonl", rows)
    pairs = load_contrast_pairs(corpus, limit=3)
    assert len(pairs) == 3


def test_load_contrast_pairs_skips_incomplete_rows(tmp_path: Path) -> None:
    row = _fixture_pair("p1")
    del row["positive"]["record"]["openui"]
    corpus = _write_fixture_corpus(tmp_path / "pairs.jsonl", [row])
    assert load_contrast_pairs(corpus) == []


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="SLM-290 corpus not present")
def test_load_contrast_pairs_reads_real_slm290_corpus_slice() -> None:
    pairs = load_contrast_pairs(REAL_CORPUS, limit=25)
    assert len(pairs) == 25
    for pair in pairs:
        assert pair.prompt
        assert pair.positive_openui != pair.negative_openui
        assert pair.family in {"content", "binding", "contract"}


# ---------------------------------------------------------------------------
# sample_contrast_pairs
# ---------------------------------------------------------------------------


def test_sample_contrast_pairs_is_deterministic_for_a_fixed_seed() -> None:
    import random

    pairs = [_make_sample(f"p{i}", family="content") for i in range(20)]
    a = sample_contrast_pairs(pairs, 5, random.Random(7))
    b = sample_contrast_pairs(pairs, 5, random.Random(7))
    assert [p.pair_id for p in a] == [p.pair_id for p in b]


def test_sample_contrast_pairs_empty_pool_raises() -> None:
    import random

    with pytest.raises(ValueError):
        sample_contrast_pairs([], 3, random.Random(0))


def test_sample_contrast_pairs_family_weights_restrict_to_named_families() -> None:
    import random

    pairs = [_make_sample(f"c{i}", family="content") for i in range(5)]
    pairs += [_make_sample(f"b{i}", family="binding") for i in range(5)]
    sample = sample_contrast_pairs(
        pairs, 20, random.Random(3), family_weights=[("content", 1.0)]
    )
    assert all(p.family == "content" for p in sample)


def _make_sample(pair_id: str, *, family: str = "content") -> ContrastPairSample:
    return ContrastPairSample(
        pair_id=pair_id,
        source_program_id=f"src_{pair_id}",
        family=family,
        transform_id=f"{family}_transform",
        severity="moderate",
        prompt="Generate a button",
        positive_openui='root = Stack([b1])\nb1 = Button(":b1.label")',
        negative_openui='root = Stack([b1])\nb1 = TextContent(":b1.label")',
    )


# ---------------------------------------------------------------------------
# pairwise_margin_contrast_loss
# ---------------------------------------------------------------------------


def test_pairwise_margin_loss_is_zero_when_positive_already_closer() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])  # identical to anchor -> distance 0
    negative = torch.tensor([[-1.0, 0.0]])  # opposite -> distance 2
    loss, pos_dist, neg_dist = pairwise_margin_contrast_loss(
        anchor, positive, negative, margin=0.2
    )
    assert float(loss) == pytest.approx(0.0)
    assert float(pos_dist[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(neg_dist[0]) == pytest.approx(2.0, abs=1e-6)


def test_pairwise_margin_loss_is_positive_when_negative_is_closer() -> None:
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[-1.0, 0.0]])  # far from anchor
    negative = torch.tensor([[1.0, 0.0]])  # identical to anchor
    loss, _, _ = pairwise_margin_contrast_loss(anchor, positive, negative, margin=0.2)
    assert float(loss) > 0.0


def test_pairwise_margin_loss_shape_mismatch_raises() -> None:
    anchor = torch.zeros(2, 4)
    positive = torch.zeros(3, 4)
    negative = torch.zeros(2, 4)
    with pytest.raises(ValueError):
        pairwise_margin_contrast_loss(anchor, positive, negative, margin=0.2)


# ---------------------------------------------------------------------------
# compute_semantic_contrast_step (orchestration)
# ---------------------------------------------------------------------------


def test_compute_semantic_contrast_step_logs_required_fields() -> None:
    pairs = [_make_sample(f"p{i}", family="content") for i in range(4)]
    pairs += [_make_sample(f"q{i}", family="binding") for i in range(4)]
    result = compute_semantic_contrast_step(
        pairs,
        _stub_rep_fn,
        objective="margin",
        margin=0.3,
        weight=0.5,
        batch_pairs=4,
        seed=11,
        corpus_path="fixture://pairs.jsonl",
        split="train",
    )
    assert isinstance(result, SemanticContrastStepResult)
    assert torch.isfinite(result.loss)
    assert result.num_pairs == 4
    assert result.loss_weight == 0.5
    assert result.margin == 0.3
    assert result.objective == "margin"
    assert result.sampling_seed == 11
    assert sum(result.family_counts.values()) == 4
    assert len(result.positive_distances) == 4
    assert len(result.negative_distances) == 4

    metrics = result.metrics_dict()
    for key in (
        "semantic_contrast_loss",
        "semantic_contrast_loss_weight",
        "semantic_contrast_objective",
        "semantic_contrast_margin",
        "semantic_contrast_pairs",
        "semantic_contrast_sampling_seed",
        "semantic_contrast_family_counts",
        "semantic_contrast_transform_counts",
        "semantic_contrast_positive_distance_mean",
        "semantic_contrast_negative_distance_mean",
    ):
        assert key in metrics, key

    report = result.to_dict()
    assert report["semantic_contrast_pair_ids"]
    assert len(report["semantic_contrast_positive_distances"]) == 4


def test_compute_semantic_contrast_step_is_deterministic_for_fixed_seed_and_step() -> None:
    pairs = [_make_sample(f"p{i}") for i in range(10)]
    kwargs = dict(
        objective="margin",
        margin=0.2,
        weight=1.0,
        batch_pairs=3,
        seed=5,
        step=2,
    )
    a = compute_semantic_contrast_step(pairs, _stub_rep_fn, **kwargs)
    b = compute_semantic_contrast_step(pairs, _stub_rep_fn, **kwargs)
    assert a.pair_ids == b.pair_ids
    assert torch.equal(a.loss, b.loss)


def test_compute_semantic_contrast_step_rejects_unsupported_objective() -> None:
    pairs = [_make_sample("p0")]
    with pytest.raises(ValueError):
        compute_semantic_contrast_step(
            pairs,
            _stub_rep_fn,
            objective="infonce",
            weight=1.0,
            batch_pairs=1,
            seed=0,
        )


def test_compute_semantic_contrast_step_rejects_non_positive_batch() -> None:
    pairs = [_make_sample("p0")]
    with pytest.raises(ValueError):
        compute_semantic_contrast_step(
            pairs, _stub_rep_fn, weight=1.0, batch_pairs=0, seed=0
        )


def test_compute_semantic_contrast_step_rejects_rep_fn_wrong_row_count() -> None:
    pairs = [_make_sample("p0")]

    def bad_rep_fn(texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts) - 1, 4)

    with pytest.raises(ValueError):
        compute_semantic_contrast_step(
            pairs, bad_rep_fn, weight=1.0, batch_pairs=1, seed=0
        )
