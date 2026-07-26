"""Block-structured bottleneck attention regression tests (AP-019 / SLM-307)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.models.block_attention import (
    SegmentKind,
    apply_loss_mask,
    build_block_bottleneck_mask,
    loss_position_mask,
)

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _segments(
    prompt: int = 2, plan: int = 3, abstract: int = 2, target: int = 3
) -> torch.Tensor:
    ids = (
        [SegmentKind.PROMPT] * prompt
        + [SegmentKind.PRIVILEGED_PLAN] * plan
        + [SegmentKind.ABSTRACT] * abstract
        + [SegmentKind.TARGET] * target
    )
    return torch.tensor([ids], dtype=torch.long)


def test_target_has_zero_direct_access_to_privileged_plan() -> None:
    segment_ids = _segments()
    mask = build_block_bottleneck_mask(segment_ids)
    blocked = torch.finfo(mask.dtype).min
    plan_start, plan_end = 2, 5
    target_start = 7
    plan_columns = mask[0, 0, target_start:, plan_start:plan_end]
    assert torch.all(plan_columns == blocked)


def test_abstract_sees_prompt_and_privileged_plan() -> None:
    segment_ids = _segments()
    mask = build_block_bottleneck_mask(segment_ids)
    # first abstract position (index 5) must see prompt(0,1), plan(2,3,4), itself(5).
    assert torch.all(mask[0, 0, 5, :6] == 0.0)


def test_target_sees_prompt_and_full_abstract_span() -> None:
    segment_ids = _segments()
    mask = build_block_bottleneck_mask(segment_ids)
    # first target position (index 7) must see prompt(0,1) and the whole abstract span(5,6).
    assert mask[0, 0, 7, 0] == 0.0
    assert mask[0, 0, 7, 1] == 0.0
    assert mask[0, 0, 7, 5] == 0.0
    assert mask[0, 0, 7, 6] == 0.0
    assert mask[0, 0, 7, 7] == 0.0


def test_causal_future_is_still_blocked() -> None:
    segment_ids = _segments()
    mask = build_block_bottleneck_mask(segment_ids)
    blocked = torch.finfo(mask.dtype).min
    assert mask[0, 0, 0, 1] == blocked  # prompt cannot see a later plan token


def test_default_off_mask_is_plain_causal_without_plan_or_target_segments() -> None:
    """Feature-off parity: an all-PROMPT sequence degrades to a plain causal mask."""
    seq_len = 5
    segment_ids = torch.full((1, seq_len), int(SegmentKind.PROMPT), dtype=torch.long)
    mask = build_block_bottleneck_mask(segment_ids)
    blocked = torch.finfo(mask.dtype).min
    expected = torch.zeros(seq_len, seq_len)
    expected[torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()] = blocked
    assert torch.equal(mask[0, 0], expected)


def test_loss_mask_only_abstract_and_target() -> None:
    segment_ids = _segments()
    keep = loss_position_mask(segment_ids)
    expected = torch.tensor(
        [[False, False, False, False, False, True, True, True, True, True]]
    )
    assert torch.equal(keep, expected)


def test_apply_loss_mask_sets_ignore_index_outside_abstract_and_target() -> None:
    segment_ids = _segments()
    labels = torch.arange(segment_ids.shape[1]).unsqueeze(0)
    masked = apply_loss_mask(labels, segment_ids, ignore_index=-100)
    assert torch.equal(masked[0, :5], torch.full((5,), -100))
    assert torch.equal(masked[0, 5:], labels[0, 5:])


def test_packed_batch_has_no_cross_example_leakage() -> None:
    segment_ids = _segments()
    example_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    mask = build_block_bottleneck_mask(segment_ids, example_ids=example_ids)
    blocked = torch.finfo(mask.dtype).min
    # example-1's abstract token (index 5) must not see example-0's prompt (index 0),
    # even though it is an earlier, causally-visible, non-plan position.
    assert mask[0, 0, 5, 0] == blocked


def _require_tiny_llama():
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModel.from_pretrained(TINY_LLAMA)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HF tiny model unavailable: {exc}")
    model.eval()
    return model


def test_bypass_canary_plan_leakage_is_fixed_by_the_mask() -> None:
    """Acceptance: 'a gradient/logit bypass canary fails before and passes
    after the mask'. Under plain causal attention (before), perturbing a
    privileged-plan position measurably moves target-position logits, since
    the plan precedes the target in sequence order. Under the block-
    bottleneck mask (after), that same perturbation must leave target logits
    unchanged.

    This fixture has no abstract span on purpose: abstract tokens are
    *meant* to carry plan content forward to the target (that mediated path
    is the whole point of the bottleneck), so a canary that includes an
    abstract span cannot isolate the direct plan->target bypass edge this
    ticket blocks. Dropping the abstract segment isolates exactly that edge.
    """
    model = _require_tiny_llama()
    torch.manual_seed(0)
    segment_ids = _segments(prompt=2, plan=2, abstract=0, target=2)
    seq_len = segment_ids.shape[1]
    input_ids = torch.randint(low=1, high=100, size=(1, seq_len))

    embed = model.get_input_embeddings()
    with torch.no_grad():
        base_embeds = embed(input_ids).clone()
        perturbed_embeds = base_embeds.clone()
        plan_positions = (segment_ids[0] == SegmentKind.PRIVILEGED_PLAN).nonzero(
            as_tuple=True
        )[0]
        perturbed_embeds[0, plan_positions] += 5.0
        target_positions = (segment_ids[0] == SegmentKind.TARGET).nonzero(
            as_tuple=True
        )[0]

        # Before: plain causal attention (no explicit mask) leaks plan -> target.
        logits_base = model(inputs_embeds=base_embeds).last_hidden_state
        logits_perturbed = model(inputs_embeds=perturbed_embeds).last_hidden_state
        delta_before = (
            (logits_perturbed[0, target_positions] - logits_base[0, target_positions])
            .abs()
            .max()
        )
        assert float(delta_before) > 1e-6, (
            "fixture invalid: plain causal attention must leak "
            "privileged-plan content into target positions before the fix"
        )

        # After: the block-bottleneck mask removes that leakage entirely.
        mask = build_block_bottleneck_mask(segment_ids, dtype=base_embeds.dtype)
        logits_base_masked = model(
            inputs_embeds=base_embeds, attention_mask=mask
        ).last_hidden_state
        logits_perturbed_masked = model(
            inputs_embeds=perturbed_embeds, attention_mask=mask
        ).last_hidden_state
        delta_after = (
            (
                logits_perturbed_masked[0, target_positions]
                - logits_base_masked[0, target_positions]
            )
            .abs()
            .max()
        )
    assert float(delta_after) < 1e-6, (
        "target hidden states must be invariant to privileged-plan content "
        "under the block-bottleneck mask"
    )
