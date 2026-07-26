"""CausalLMOpenUIPlugin.forward_with_segments wiring tests (AP-019 / SLM-307)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from slm_training.models.block_attention import SegmentKind
from slm_training.models.causal_lm_openui import (
    CausalLMOpenUIConfig,
    CausalLMOpenUIPlugin,
)

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _require_plugin() -> CausalLMOpenUIPlugin:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA)
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny llama fixture unavailable: {exc}")
    config = CausalLMOpenUIConfig(base_model_id=TINY_LLAMA, base_model_revision="main")
    return CausalLMOpenUIPlugin(model, tokenizer, config)


def _segment_batch(
    prompt: int = 2, plan: int = 2, abstract: int = 2, target: int = 2
) -> torch.Tensor:
    ids = (
        [SegmentKind.PROMPT] * prompt
        + [SegmentKind.PRIVILEGED_PLAN] * plan
        + [SegmentKind.ABSTRACT] * abstract
        + [SegmentKind.TARGET] * target
    )
    return torch.tensor([ids], dtype=torch.long)


def test_forward_with_segments_counts_only_abstract_and_target_tokens() -> None:
    plugin = _require_plugin()
    torch.manual_seed(0)
    segment_ids = _segment_batch()
    vocab_size = plugin.model.config.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, segment_ids.shape[1]))

    result = plugin.forward_with_segments(input_ids, segment_ids)

    assert result["effective_token_count"] == 4  # 2 abstract + 2 target
    assert result["loss"] == result["loss"]  # not NaN


def test_forward_with_segments_blocks_plan_leakage_into_target_logits() -> None:
    """No abstract span on purpose: it is the direct plan->target bypass edge
    under test, not the legitimate abstract-mediated path (see
    test_block_attention.py::test_bypass_canary_plan_leakage_is_fixed_by_the_mask).
    """
    plugin = _require_plugin()
    torch.manual_seed(0)
    segment_ids = _segment_batch(abstract=0)
    vocab_size = plugin.model.config.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, segment_ids.shape[1]))

    from slm_training.models.block_attention import build_block_bottleneck_mask

    embed = plugin.model.get_input_embeddings()
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

        mask = build_block_bottleneck_mask(segment_ids, dtype=base_embeds.dtype)
        logits_base = plugin.model(
            inputs_embeds=base_embeds, attention_mask=mask
        ).logits
        logits_perturbed = plugin.model(
            inputs_embeds=perturbed_embeds, attention_mask=mask
        ).logits
        delta = (
            (logits_perturbed[0, target_positions] - logits_base[0, target_positions])
            .abs()
            .max()
        )
    assert float(delta) < 1e-6


def test_forward_with_segments_does_not_change_legacy_forward() -> None:
    """The new opt-in method must not touch the existing full-sequence path."""
    import inspect

    from slm_training.models.causal_lm_openui import CausalLMOpenUIPlugin

    source = inspect.getsource(CausalLMOpenUIPlugin.forward)
    assert "forward_with_segments" not in source
    assert "block_attention" not in source
