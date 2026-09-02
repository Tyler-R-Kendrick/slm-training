"""S12/N3: per-step commit-authority histogram for the MaskGIT decode loop.

Records, for each denoising step, how many canvas positions were committed and
under which authority -- ``forced`` (a DFA force-emit proof, I2), ``confident``
(the model's constrained argmax over the legal set), or ``speculative`` (an I3
speculative-rank commit). Opt-in: a decode that does not arm the flag must be
byte-identical to one that never had the telemetry.
"""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import DecodeStats, collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":slot_0")'
AUTHORITIES = ("forced", "confident", "speculative")


def _tiny_model(**overrides) -> TwoTowerModel:
    records = [
        ExampleRecord(id="a", prompt="Hero card", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="Call to action", openui=CTA, split="train"),
    ]
    cfg = TwoTowerConfig(
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=4,
        seed=0,
        grammar_ltr_primary=False,  # positionwise MaskGIT is the recorded path
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    model.eval()
    return model


def _digest(texts: list[str]) -> str:
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()


def test_telemetry_is_off_by_default() -> None:
    stats = DecodeStats()
    assert stats.record_step_commits is False
    assert stats.step_commits == []


def test_off_by_default_costs_nothing_and_stays_byte_identical() -> None:
    model = _tiny_model()
    prompts = ["Hero card", "Call to action"]

    quiet = DecodeStats()
    with collect_decode_stats(quiet):
        without = model.generate_batch(prompts)

    loud = DecodeStats()
    loud.record_step_commits = True
    with collect_decode_stats(loud):
        with_telemetry = model.generate_batch(prompts)

    assert _digest(without) == _digest(with_telemetry)
    assert quiet.step_commits == []
    assert loud.step_commits, "armed telemetry recorded no step"
    # Recording is pure bookkeeping: it must not change the decode's work.
    assert quiet.forwards_count == loud.forwards_count
    assert quiet.tokens_emitted == loud.tokens_emitted


@pytest.mark.parametrize("mode", ["topk", "confidence", "adaptive"])
def test_every_step_row_partitions_its_commits_by_authority(mode: str) -> None:
    model = _tiny_model(parallel_unmask=mode)
    stats = DecodeStats()
    stats.record_step_commits = True
    with collect_decode_stats(stats):
        model.generate_batch(["Hero card", "Call to action"])

    assert stats.step_commits
    for row in stats.step_commits:
        assert set(row) == {
            "step",
            "committed",
            "forced",
            "confident",
            "speculative",
            "forwards",
            "authority",
        }
        assert row["committed"] >= 1
        assert row["forced"] + row["confident"] + row["speculative"] == row["committed"]
        assert min(row["forced"], row["confident"], row["speculative"]) >= 0
        assert row["forwards"] >= 0
        assert row["authority"] in AUTHORITIES
        assert row[str(row["authority"])] == max(row[name] for name in AUTHORITIES)


EXACT_SOURCE = 'root = Card([title])\ntitle = TextContent(":slot_0")\n'


def test_singleton_bypass_steps_are_recorded_as_zero_forward_forced_steps() -> None:
    """The I2 one-hole bypass is a forced step that spent no forward.

    Seeds the canvas from a certified program with a single ``=`` masked, so
    the DFA proves the hole and ``_generate_maskgit_one`` takes the
    ``exact_commit`` branch. The recorded row must say so: one commit, forced
    authority, ``forwards == 0``.
    """
    model = _tiny_model(
        gen_steps=1,
        grammar_fastpath_mode="ltr",
        structural_bias=1_000.0,
        allow_unconstrained_fallback=False,
    )
    tokenizer = model.tokenizer
    seed = [
        tokenizer.bos_id,
        *tokenizer.encode(EXACT_SOURCE, add_special=False),
        tokenizer.eos_id,
    ]
    equals = [
        index
        for index, token_id in enumerate(seed)
        if token_id == tokenizer.token_to_id["="]
    ]
    seed[equals[0]] = tokenizer.mask_id
    ctx, ctx_pad = model._encode_context(["Hero card"])

    stats = DecodeStats()
    stats.record_step_commits = True
    with collect_decode_stats(stats):
        model._generate_maskgit_one(
            ctx,
            ctx_pad,
            len(seed),
            use_grammar=True,
            slot_contract=[":slot_0"],
            seed_ids=seed,
        )

    assert stats.forced_tokens == 1
    assert stats.step_commits == [
        {
            "step": 0,
            "committed": 1,
            "forced": 1,
            "confident": 0,
            "speculative": 0,
            "forwards": 0,
            "authority": "forced",
        }
    ]


def test_zero_forward_steps_are_always_forced_steps() -> None:
    """A step that committed without a forward can only have been proven."""
    model = _tiny_model()
    stats = DecodeStats()
    stats.record_step_commits = True
    with collect_decode_stats(stats):
        model.generate_batch(["Hero card", "Call to action"])
    for row in stats.step_commits:
        if row["forwards"] == 0:
            assert row["forced"] == row["committed"]
            assert row["authority"] == "forced"


def test_merge_keeps_the_switch_a_switch_and_the_history_per_call() -> None:
    """``bool`` is an ``int`` subclass; summing the flag would corrupt it."""
    left = DecodeStats()
    left.record_step_commits = True
    left.step_commits.append({"step": 0, "committed": 1})
    right = DecodeStats()
    right.record_step_commits = True
    right.step_commits.append({"step": 0, "committed": 2})
    left.merge(right)
    assert left.record_step_commits is True
    assert len(left.step_commits) == 1


def test_round_trips_through_as_dict_and_from_dict() -> None:
    stats = DecodeStats()
    stats.record_step_commits = True
    stats.step_commits.append(
        {
            "step": 0,
            "committed": 3,
            "forced": 1,
            "confident": 2,
            "speculative": 0,
            "forwards": 1,
            "authority": "confident",
        }
    )
    restored = DecodeStats.from_dict(stats.as_dict())
    assert restored.record_step_commits is True
    assert restored.step_commits == stats.step_commits
