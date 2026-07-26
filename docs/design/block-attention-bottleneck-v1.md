# Block-structured bottleneck attention (AP-019 / SLM-307)

`slm_training.models.block_attention` implements the core Abstract-CoT
bottleneck (arXiv:2604.22709): a sequence is split into four ordered
segments — `prompt`, `privileged_plan`, `abstract` (the reasoning
bottleneck), and `target` (the answer/program) — and `target` positions may
**never** attend `privileged_plan` positions directly, even though a plain
causal mask would otherwise expose them (the plan precedes the target in
sequence order).

This is mask/loss-mechanism plumbing only. No production training path calls
it by default, and no privileged-plan-augmented record format exists in the
repository yet — that data pipeline (and the reasoning/distillation harness
wiring the issue also names as in-scope) is deferred to the follow-on
AP-020/AP-021 work that actually produces such records, so it can build on a
tested, correct primitive rather than a placeholder.

## Mask semantics

`build_block_bottleneck_mask(segment_ids, example_ids=None)` returns an
additive `[batch, 1, seq, seq]` float mask, following the same convention
`models/hf_denoiser.py::_bidirectional_mask` already uses (passed as
`attention_mask=` to override a HF backbone's internal causal mask).

Visibility is plain causal (query `i` may see key `j` only when `j <= i`)
with exactly one additional restriction: a `TARGET` query may never see a
`PRIVILEGED_PLAN` key, regardless of causal order. Every other documented
rule falls out of causal order alone, since the four segments always appear
in that fixed order:

* abstract tokens attend prompt + privileged plan + prior abstract (causal
  already blocks abstract from seeing the not-yet-generated target);
* target tokens attend prompt + full abstract span + prior targets, **never**
  privileged-plan positions (the one added restriction above).

`example_ids` (optional, `[batch, seq]`) additionally blocks any query from
attending a key in a different packed example, preventing cross-example
leakage inside one packed row.

`loss_position_mask(segment_ids)` / `apply_loss_mask(labels, segment_ids)`
implement "only abstract and target positions contribute to loss" by setting
every other label to `ignore_index` (`-100`), matching the ordinary HF
cross-entropy `ignore_index` convention.

## Bypass canary

The acceptance criterion "a gradient/logit bypass canary fails before and
passes after the mask" is `test_block_attention.py::test_bypass_canary_plan_leakage_is_fixed_by_the_mask`.
It uses a `prompt + privileged_plan + target` fixture **with no abstract
span** on purpose: abstract tokens are *meant* to carry plan content forward
to the target (that mediated path is the entire point of the bottleneck), so
a canary that includes an abstract span cannot isolate the direct
`plan -> target` bypass edge this ticket blocks. Perturbing the plan
embedding then measurably moves target hidden states under plain causal
attention (no mask — the pre-fix state) and leaves them exactly unchanged
under `build_block_bottleneck_mask` (the post-fix state).

## Integration point

`CausalLMOpenUIPlugin.forward_with_segments(input_ids, segment_ids,
example_ids=None)` (`src/slm_training/models/causal_lm_openui.py`) is a new,
separate, opt-in method that builds the mask and masked labels and calls the
model directly via `inputs_embeds`/`attention_mask`/`labels`, mirroring
`HFDenoiserTower`'s established pattern. It also returns
`effective_token_count` (the number of abstract+target positions actually
contributing to loss), satisfying "log effective token counts."

`CausalLMOpenUIPlugin.forward()` — the existing full-sequence SFT loss path
— is completely untouched:
`test_causal_lm_openui_block_attention.py::test_forward_with_segments_does_not_change_legacy_forward`
asserts its source never references the new method or module. The feature is
stable and default-off simply because nothing calls the new method unless a
caller explicitly builds segment ids and invokes it.

## Reproduction

```bash
pytest -q tests/test_models/test_block_attention.py tests/test_models/test_causal_lm_openui_block_attention.py
```

Both suites are plain unit tests (the HF-model-backed cases use the tiny
`hf-internal-testing/tiny-random-LlamaForCausalLM` fixture already used by
`test_hf_denoiser.py`) and complete in seconds, well inside the repository's
hard run cap (AGENTS.md § "Hard run cap").
