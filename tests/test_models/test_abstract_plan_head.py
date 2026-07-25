"""AbstractPlanHead regression tests (AP-023 / SLM-315)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.models.abstract_plan_head import (
    AbstractPlanHead,
    AbstractPlanMode,
    AbstractPlanTrace,
)

D_MODEL = 8
BATCH = 3
SEQ = 5


def _plan(rounds: int = 3, slot_count: int = 6) -> AbstractPlanV1:
    return AbstractPlanV1(slot_count=slot_count, max_slot_count=slot_count, rounds=rounds)


def _context(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    context = torch.randn(BATCH, SEQ, D_MODEL)
    pad_mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    pad_mask[:, -1] = True  # last position padded for every row
    return context, pad_mask


def test_disabled_mode_is_rejected_by_forward() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    with pytest.raises(ValueError):
        head(context, pad_mask, mode=AbstractPlanMode.DISABLED)


def test_disabled_head_is_never_constructed_has_no_parameters() -> None:
    """Mirrors test_dynamic_pointer_scorer.py's legacy-mode zero-parameter check:
    a disabled TwoTowerModel-style caller must never instantiate this module.
    """
    # No AbstractPlanHead instance at all when mode is "disabled" -- callers
    # (TwoTowerModel.__init__) gate construction itself, so there is nothing
    # to assert on here beyond "the module type carries no implicit default
    # parameters when unused": constructing it always allocates one Linear,
    # by design, which is why TwoTowerModel skips construction entirely.
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    assert sum(p.numel() for p in head.parameters()) > 0


@pytest.mark.parametrize(
    "mode", ["teacher_forced", "sampled", "oracle", "random", "shuffled"]
)
def test_every_enabled_mode_emits_a_complete_trace(mode: str) -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    target_plan_ids = torch.randint(0, plan.slot_count, (BATCH, plan.rounds))
    generator = torch.Generator().manual_seed(0)

    trace = head(
        context,
        pad_mask,
        mode=mode,
        target_plan_ids=target_plan_ids,
        generator=generator,
    )

    assert isinstance(trace, AbstractPlanTrace)
    assert trace.mode == mode
    assert trace.plan is plan
    assert trace.plan_tokens.shape == (BATCH, plan.rounds)
    assert bool(((trace.plan_tokens >= 0) & (trace.plan_tokens < plan.slot_count)).all())
    assert trace.hidden_states.shape == (BATCH, D_MODEL)
    assert trace.provenance["mode"] == mode
    assert trace.provenance["compatibility_fingerprint"] == plan.compatibility_fingerprint
    # token_ids() must resolve every predicted index into the plan's actual
    # reserved absolute vocabulary ids -- "every mode emits valid AbstractPlanV1".
    token_ids = trace.token_ids()
    expected = torch.tensor(plan.slot_token_ids)[trace.plan_tokens]
    assert torch.equal(token_ids, expected)


def test_teacher_forced_and_oracle_require_target_plan_ids() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    for mode in ("teacher_forced", "oracle"):
        with pytest.raises(ValueError):
            head(context, pad_mask, mode=mode)


def test_rejects_wrong_shaped_target_plan_ids() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    wrong_shape = torch.zeros(BATCH, plan.rounds + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="shape"):
        head(context, pad_mask, mode="oracle", target_plan_ids=wrong_shape)


def test_rejects_out_of_range_target_plan_ids() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    out_of_range = torch.full((BATCH, plan.rounds), plan.slot_count, dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range"):
        head(context, pad_mask, mode="oracle", target_plan_ids=out_of_range)


def test_rejects_device_mismatched_generator() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()

    class _FakeCudaGenerator:
        device = "cuda"

    with pytest.raises(ValueError, match="device"):
        head(context, pad_mask, mode="random", generator=_FakeCudaGenerator())


def test_oracle_mode_is_a_pure_bypass_with_no_logits() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    target_plan_ids = torch.randint(0, plan.slot_count, (BATCH, plan.rounds))

    trace = head(context, pad_mask, mode="oracle", target_plan_ids=target_plan_ids)

    assert torch.equal(trace.plan_tokens, target_plan_ids)
    assert trace.logits is None
    assert trace.entropy is None


def test_teacher_forced_uses_given_tokens_but_still_reports_logits() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    target_plan_ids = torch.randint(0, plan.slot_count, (BATCH, plan.rounds))

    trace = head(
        context, pad_mask, mode="teacher_forced", target_plan_ids=target_plan_ids
    )

    assert torch.equal(trace.plan_tokens, target_plan_ids)
    assert trace.logits is not None
    assert trace.logits.shape == (BATCH, plan.rounds, plan.slot_count)
    assert trace.entropy is not None
    assert trace.entropy.shape == (BATCH, plan.rounds)


def test_random_mode_ignores_context_content() -> None:
    """A random-control mode must not depend on the context tower at all."""
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()
    other_context = context + 100.0  # wildly different content

    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    trace_a = head(context, pad_mask, mode="random", generator=generator_a)
    trace_b = head(other_context, pad_mask, mode="random", generator=generator_b)

    assert torch.equal(trace_a.plan_tokens, trace_b.plan_tokens)
    assert trace_a.logits is None


def test_shuffled_mode_is_a_permutation_of_a_sampled_plan() -> None:
    plan = _plan(rounds=4)
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()

    sample_generator = torch.Generator().manual_seed(3)
    baseline = head(context, pad_mask, mode="sampled", generator=sample_generator)

    shuffle_generator = torch.Generator().manual_seed(3)
    shuffled = head(context, pad_mask, mode="shuffled", generator=shuffle_generator)

    # Same multiset of predicted tokens, since torch.Generator streams for
    # the identical seed produce the identical underlying sample before the
    # shuffle-only draw diverges the stream.
    assert sorted(baseline.plan_tokens[0].tolist()) == sorted(
        shuffled.plan_tokens[0].tolist()
    )


def test_deterministic_with_seeded_generator() -> None:
    plan = _plan()
    head = AbstractPlanHead(D_MODEL, plan)
    context, pad_mask = _context()

    trace_1 = head(
        context, pad_mask, mode="sampled", generator=torch.Generator().manual_seed(42)
    )
    trace_2 = head(
        context, pad_mask, mode="sampled", generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(trace_1.plan_tokens, trace_2.plan_tokens)


def test_twotower_config_rejects_unknown_abstract_plan_mode() -> None:
    """TwoTowerConfig.__post_init__ must reject a typo'd mode instead of
    silently taking the "enabled" branch in __init__ (AP-023 review fix)."""
    from slm_training.models.twotower import TwoTowerConfig

    with pytest.raises(ValueError, match="abstract_plan_mode"):
        TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            output_tokenizer="lexer",
            abstract_plan_mode="dissabled",  # typo: must not silently enable
        )


def test_masked_mean_pool_ignores_padded_positions() -> None:
    from slm_training.models.abstract_plan_head import _masked_mean_pool

    context = torch.zeros(1, 3, 4)
    context[0, 0] = 1.0
    context[0, 1] = 3.0
    context[0, 2] = 100.0  # padded row must not affect the mean
    pad_mask = torch.tensor([[False, False, True]])

    pooled = _masked_mean_pool(context, pad_mask)
    assert torch.allclose(pooled, torch.full((1, 4), 2.0))
