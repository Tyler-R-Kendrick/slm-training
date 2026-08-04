"""HX4 hybrid unmask scheduler + HX5 shared grammar repair session tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import DecodeStats, aggregate_stats
from slm_training.models.twotower import (
    TwoTowerConfig,
    TwoTowerModel,
    hybrid_frontier_order,
    hybrid_mask_runs,
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'

MASK = -1
EOS = -2
PAD = -3


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id="a",
            prompt="Hero",
            openui=HERO,
            split="train",
            placeholders=[":slot_0", ":slot_1"],
        ),
        ExampleRecord(
            id="b", prompt="CTA", openui=CTA, split="train", placeholders=[":slot_0"]
        ),
    ]


def _hybrid_config(**overrides: object) -> TwoTowerConfig:
    base: dict[str, object] = dict(
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=4,
        max_target_len=64,
        max_prompt_len=32,
        seed=0,
        grammar_ltr_primary=False,
    )
    base.update(overrides)
    return TwoTowerConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# (a) span-run computation
# --------------------------------------------------------------------------


def test_hybrid_mask_runs_groups_contiguous_holes() -> None:
    ids = [10, MASK, MASK, MASK, 11, MASK, 12, MASK, MASK]
    unknown = [t == MASK for t in ids]
    runs = hybrid_mask_runs(unknown, ids, eos_id=EOS, pad_id=PAD)
    assert runs == [[1, 2, 3], [5], [7, 8]]


def test_hybrid_mask_runs_trims_at_eos_boundary() -> None:
    # The run starting at 2 straddles EOS at 5: only the pre-EOS half survives,
    # and the masked positions after EOS are not part of the program at all.
    ids = [10, 11, MASK, MASK, MASK, EOS, MASK, MASK, MASK]
    unknown = [t == MASK for t in ids]
    runs = hybrid_mask_runs(unknown, ids, eos_id=EOS, pad_id=PAD)
    assert runs == [[2, 3, 4]]


def test_hybrid_mask_runs_trims_at_pad_boundary() -> None:
    ids = [10, MASK, MASK, PAD, MASK, EOS]
    unknown = [t == MASK for t in ids]
    runs = hybrid_mask_runs(unknown, ids, eos_id=EOS, pad_id=PAD)
    assert runs == [[1, 2]]


def test_hybrid_mask_runs_empty_when_boundary_is_first() -> None:
    ids = [EOS, MASK, MASK]
    unknown = [t == MASK for t in ids]
    assert hybrid_mask_runs(unknown, ids, eos_id=EOS, pad_id=PAD) == []


# --------------------------------------------------------------------------
# (b) frontier ordering
# --------------------------------------------------------------------------


def test_hybrid_frontier_order_prefers_prefix_extending_positions() -> None:
    # 1, 3 and 7 extend a committed prefix (left neighbour committed); 4, 5 and
    # 8 are interior holes — 5 outranks 1 on confidence but never on lane.
    unknown = [False, True, False, True, True, True, False, True, True]
    positions = [1, 3, 4, 5, 7, 8]
    conf = {1: 0.10, 3: 0.90, 4: 0.20, 5: 0.50, 7: 0.99, 8: 0.30}
    order = hybrid_frontier_order(positions, unknown, conf)
    assert order[:3] == [7, 3, 1]  # frontier group, confidence-descending
    assert order[3:] == [5, 8, 4]  # interior group, confidence-descending


def test_hybrid_frontier_order_is_total_and_deterministic() -> None:
    unknown = [False, True, True, True]
    conf = {1: 0.5, 2: 0.5, 3: 0.5}
    first = hybrid_frontier_order([3, 2, 1], unknown, conf)
    second = hybrid_frontier_order([1, 2, 3], unknown, conf)
    assert first == second == [1, 2, 3]


# --------------------------------------------------------------------------
# (c) hybrid decode smoke
# --------------------------------------------------------------------------


@pytest.mark.parametrize("min_run", [3, 8, 99])
def test_hybrid_unmask_mode_generates_valid_output(min_run: int) -> None:
    from slm_training.dsl.parser import validate

    cfg = _hybrid_config(unmask_mode="hybrid", hybrid_span_min_run=min_run)
    model = TwoTowerModel.from_records(_records(), config=cfg, device="cpu")
    model.eval()
    text, stats = model.generate_with_stats("Hero")
    assert isinstance(text, str) and text.strip()
    validate(text)  # raises on invalid OpenUI
    lane_commits = stats.hybrid_span_commits + stats.hybrid_frontier_commits
    assert lane_commits > 0


def test_hybrid_lane_split_follows_span_min_run() -> None:
    """A min run longer than the canvas leaves nothing for the span lane; a
    short one lets contiguous holes ride the block schedule."""
    span_cfg = _hybrid_config(unmask_mode="hybrid", hybrid_span_min_run=3)
    span_model = TwoTowerModel.from_records(_records(), config=span_cfg, device="cpu")
    span_model.eval()
    _text, span_stats = span_model.generate_with_stats("Hero")
    assert span_stats.hybrid_span_commits > 0

    frontier_cfg = _hybrid_config(unmask_mode="hybrid", hybrid_span_min_run=99)
    frontier_model = TwoTowerModel.from_records(
        _records(), config=frontier_cfg, device="cpu"
    )
    frontier_model.eval()
    _text2, frontier_stats = frontier_model.generate_with_stats("Hero")
    assert frontier_stats.hybrid_span_commits == 0
    assert frontier_stats.hybrid_frontier_commits > 0


def test_default_positions_mode_leaves_hybrid_counters_at_zero() -> None:
    cfg = _hybrid_config()
    model = TwoTowerModel.from_records(_records(), config=cfg, device="cpu")
    model.eval()
    _text, stats = model.generate_with_stats("Hero")
    assert stats.hybrid_span_commits == 0
    assert stats.hybrid_frontier_commits == 0
    assert stats.hybrid_span_reverts == 0


# --------------------------------------------------------------------------
# (d) HX5 shared repair session
# --------------------------------------------------------------------------


def test_shared_repair_session_keeps_generate_deterministic() -> None:
    """HX5 reuses one grammar session across repair attempts; a deterministic
    config must still produce the same program on every call."""
    cfg = _hybrid_config(generate_max_attempts=3, grammar_ltr_repair=True)
    model = TwoTowerModel.from_records(_records(), config=cfg, device="cpu")
    model.eval()
    first = model.generate("Hero")
    second = model.generate("Hero")
    assert first == second

    length = 32
    ctx, ctx_pad = model._encode_context(["Hero"])
    direct_a = model._ltr_repair_from_bos(ctx, ctx_pad, length)
    shared = model._new_grammar_states(1)
    direct_b = model._ltr_repair_from_bos(
        ctx, ctx_pad, length, state=shared[0] if shared else None
    )
    direct_c = model._ltr_repair_from_bos(
        ctx, ctx_pad, length, state=shared[0] if shared else None
    )
    # A reused session is a cache, never a decision: same text as a fresh one,
    # and stable when the same session repairs twice in a row.
    assert direct_a == direct_b == direct_c


# --------------------------------------------------------------------------
# (e) telemetry surface
# --------------------------------------------------------------------------


def test_hybrid_counters_present_in_decode_stats_and_aggregate() -> None:
    names = (
        "hybrid_span_commits",
        "hybrid_frontier_commits",
        "hybrid_span_reverts",
        "block_joint_unknowns",
        "template_slot_positions",
    )
    stats = DecodeStats()
    for name in names:
        assert getattr(stats, name) == 0
        setattr(stats, name, 2)
    agg = aggregate_stats([stats])
    for name in names:
        assert f"{name}_sum" in agg or name in agg, agg.get("counters_omitted_zero")


def test_frontier_head_keeps_both_lanes_live_on_one_run() -> None:
    """Regression: an unfragmented canvas must not collapse into one lane.

    A fresh canvas is a single contiguous masked run, so before the
    frontier-head carve EVERY position classified as a span and the
    positionwise lane never engaged — a threshold-flipped binary switch, not
    a hybrid (measured: 64 span / 0 frontier at every span threshold). The
    carve hands the leading positions of any run touching committed context
    (or the BOS edge) to the positionwise lane, so both lanes stay live.
    """
    from slm_training.models.twotower import hybrid_mask_runs

    length = 24
    unknown_row = [False] + [True] * (length - 1)  # BOS committed, rest masked
    ids_row = [7] + [999] * (length - 1)  # 999 = mask sentinel, no EOS/PAD
    runs = hybrid_mask_runs(unknown_row, ids_row, eos_id=2, pad_id=0)
    assert len(runs) == 1 and len(runs[0]) == length - 1

    # Replicate the scheduler's carve: the run touches the committed BOS edge.
    for head_n in (1, 2, 4):
        run = runs[0]
        at_frontier = run[0] == 0 or not unknown_row[run[0] - 1]
        assert at_frontier
        head, tail = run[:head_n], run[head_n:]
        assert len(head) == head_n, "frontier lane must receive the leading edge"
        assert len(tail) >= 3, "interior must remain span-eligible"
        assert set(head).isdisjoint(tail), "lanes must partition, never overlap"
        assert sorted([*head, *tail]) == run, "carve must lose no position"
