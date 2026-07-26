# AP-024 (SLM-316): abstract-plan denoiser connector

Status: harness wiring only. **No training run or benchmark was executed as
part of this change** -- this document registers a connector, its wiring, and
its regression tests, not an experiment result. In particular this is not a
claim that conditioning on the plan helps generation quality; that is AP-022's
(SLM-313) job, once this wiring and AP-021's warm-up loop exist.

## Decision

Let the existing MaskGIT denoiser consume the AP-023 (SLM-315) discrete
abstract plan through explicit conditioning while retaining prompt context,
grammar state, and explicit token/tree state -- per arXiv:2604.22709.

## What this adds

`src/slm_training/models/abstract_plan_connector.py`:

- **`PlanConnectorArm`** -- seven named arms sharing one code path:
  `disabled`, `learned`/`oracle` (the head's actual predicted plan, gradient
  connected), `detached` (same content, gradient blocked via `.detach()`,
  mirroring `SharedRecursiveDenoiserTower`'s `detach_between_steps` arm H
  convention), and three content-null controls at matched architecture --
  `empty` (all-zero vector), `random` (fresh random vector, seedable),
  `shuffled` (another example's plan vector via a batch-dimension
  permutation). `resolve_plan_vector(vector, arm=..., generator=...)` is the
  single function every arm resolves through, so a control arm cannot
  silently diverge from `learned` in shape, device, or dtype.
- **`AbstractPlanConnector`** (`nn.Module`) -- owns a small
  `plan_slot_count`-row embedding table for the plan's *codebook-local* slot
  indices (`AbstractPlanTrace.plan_tokens`, not the absolute reserved
  vocabulary ids `AbstractPlanV1.token_ids()` uses, which are reserved in the
  causal-LM tokenizer's namespace per AP-016/017, not necessarily valid rows
  in an arbitrary TwoTower vocabulary), a learnable scalar `gate` initialized
  at `0.0`, and a `Linear(d_model, vocab_size)` projection. `bias_for_vocab`
  computes `tanh(gate) * proj(plan_vector)`, gathered to `candidate_ids` when
  given.
- **`PlanConnectorTrace`** -- one `DenoiserTower.project` call's worth of
  evidence: `gate_value`, `bias_norm`, and a correlational
  `choice_changed_count` / `position_count` (did the argmax token flip when
  the bias was added) -- explicitly not a causal-use claim, matching
  `CausalLatentUseFalsificationSpecV1` in `models/causal_trace.py`.

## Where it's wired, and why that's the safe choice

The repository already has an established idiom for conditioning generation:
a **post-hoc additive bias on already-computed vocabulary logits**, not a
change to the transformer stack itself --
`DenoiserTower._runtime_symbol_features` (`models/blocks.py`) and
`TwoTowerModel._component_inventory_bias`/`_component_plan_bias`/etc.
(`models/twotower.py`) all follow this shape, each pairing a
`*_applications`/`*_choice_changes` counter with the bias addition.

This connector follows the same idiom, wired into
`DenoiserTower.project` (`models/blocks.py`) rather than into the MaskGIT
round loop (`TwoTowerModel._generate_maskgit_one`, an 800+ line method
handling grammar fastpaths, speculative decoding, and singleton-bypass
shortcuts) or the transformer layer stack (`DenoiserTower.encode`). Because
every MaskGIT round already calls `project` to turn hidden states into the
logits that drive that round's unmask/remask decisions, attaching the bias
there means:

- **zero changes to the round loop's scheduling logic** -- "thread plan
  context through every refinement round" falls out of `project` being
  called every round, not from a new argument threaded through hundreds of
  call sites;
- **the singleton/exact-forced-token bypass paths are untouched** -- they
  never call `project` at all, so the deterministic bypass (decode invariant
  I2) cannot be perturbed by this connector, by construction;
- **per-round choice-change instrumentation is nearly free** -- two argmax
  calls on already-computed logits, not a second forward pass;
- **LTR/compiler-candidate scoring is correctly left alone** -- those call
  sites project single-row (`[D]`) or per-candidate slices, not the
  batch-shaped `[B, T, vocab]` tensor the MaskGIT round loop produces;
  `_apply_plan_connector_bias` explicitly skips non-3D logits rather than
  guessing how to reshape them.

`DenoiserTower.set_plan_connector(connector, arm=...)` /
`set_plan_vector(vector)` / `pop_plan_connector_traces()` mirror the existing
`set_runtime_symbol_features`/`_features_for_batch` mutable-attribute
convention: `None` is the default, checked before any new code path runs, so
disabled is a true zero-parameter, zero-compute no-op -- not merely a
zero-valued one.

`TwoTowerModel.abstract_plan_connector` is a **property** reading
`denoiser._plan_connector`, not a second submodule attribute of its own.
Registering the same `nn.Module` instance under two attribute paths
(`self.abstract_plan_connector` *and* `self.denoiser._plan_connector`) does
not raise, but `named_parameters()`'s identity-based deduplication silently
keeps only the first-visited path -- which was `denoiser._plan_connector.*`,
since `self.denoiser` is constructed long before this connector in
`__init__`. An earlier version of this change registered both, which made
`optimizer_parameter_groups`'s prefix match nothing; the property fixes the
double registration, and the optimizer prefix now reads
`"denoiser._plan_connector."`, the real path.

## Deferred, honestly

- **Tree/recursive denoising.** `SharedRecursiveDenoiserTower`
  (`models/recursive_denoiser.py`) duck-types `DenoiserTower`'s public
  contract but does not implement `set_plan_connector`/the `project` bias
  hook. `TwoTowerConfig.__post_init__` rejects
  `abstract_plan_connector_arm != "disabled"` combined with
  `denoiser_arch="shared_recursive"` with a clear `ValueError` rather than
  silently no-op'ing or crashing with `AttributeError` at generation time.
  Wiring the tree denoiser is follow-on work, not represented as done here.
- **HF-backed denoising.** `HFDenoiserTower` (`models/hf_denoiser.py`) is a
  standalone `nn.Module` -- not a `DenoiserTower` subclass -- with its own
  `project()` and no plan-connector hook at all. `__post_init__` rejects
  `abstract_plan_connector_arm != "disabled"` combined with
  `denoiser_backend in {"hf", "huggingface", "transformers"}` the same way.
  `StackedMatchedStateDenoiserTower` (`models/recursive_denoiser.py`,
  `denoiser_arch="stacked_matched_state"`) is deliberately *not* rejected:
  it subclasses `DenoiserTower` without overriding `project()`, so it
  inherits the hook correctly and needs no guard.
- **`GrammarDiffusionModel`** (`models/grammar_diffusion.py`) is an
  independent model/decoder with its own `GrammarDenoiser` and its own
  per-phase refinement loop; it is not touched by this change.
- **`semantic_connector.py`**'s three connector variants (`Linear`,
  `LowRank`, `CrossAttention`) are a separate, still-unwired factory
  (`TwoTowerConfig.semantic_connector`); this change does not use or modify
  them. This connector's own architecture (gated-additive logit bias) was
  chosen because it matches the codebase's existing conditioning idiom
  exactly and composes safely with the MaskGIT round loop without touching
  it, not because the alternative cross-attention shape was evaluated and
  rejected.

## Acceptance criteria mapping

- "Add one preregistered connector: cross-attention or gated additive
  conditioning." -- one connector (gated-additive), matching the codebase's
  existing logit-bias idiom.
- "Thread plan context through every refinement round." -- `set_plan_vector`
  persists across every `project` call until cleared; every MaskGIT round
  calls `project`.
- "Support oracle, learned, empty, random, shuffled, detached-plan, and
  disabled arms through one code path." -- `PlanConnectorArm` /
  `resolve_plan_vector`.
- "Off mode remains bit-exact." --
  `test_abstract_plan_connector_is_bit_exact_with_baseline_when_disabled`:
  identical parameters and identical `training_loss` under matched seeds
  between a baseline model and one with `abstract_plan_connector_arm`
  explicitly `"disabled"`.
- "All control arms share initialization, prompt, noise/mask schedule, and
  seed." -- `generate_with_plan_connector` delegates to the existing,
  unmodified `generate()`; only the plan vector fed to the denoiser differs
  by arm.
- "Trace identifies plan-conditioned decision changes by round and
  position." -- `pop_plan_connector_traces()` returns one `PlanConnectorTrace`
  per `project` call (one per round), each carrying a per-call
  `choice_changed_count`/`position_count`.
- "No production default change." -- `abstract_plan_connector_arm` defaults
  to `"disabled"`; ordinary `training_loss`/`forward` never call the
  connector. SLM-313's explicit opt-in training contract is the sole
  exception and requires a positive plan loss plus enabled head/connector.

## Tests

`tests/test_models/test_abstract_plan_connector.py` -- pure-tensor arm
resolution (passthrough, detach, empty, random, shuffled, disabled-rejects,
generator-device mismatch), `plan_vector_from_trace` pooling, `bias_for_vocab`
candidate-gather equivalence, the zero-gate no-op property, and
`DenoiserTower.project` wiring (bit-exact when no connector/vector attached,
bit-exact again after clearing, one trace per call with a bounded
choice-change count, batch-size mismatch raises, non-3D hidden is skipped).

`tests/test_harnesses/model_build/test_twotower.py` additions -- disabled by
default and `None`; config validation (`abstract_plan_connector_arm` requires
`abstract_plan_mode != "disabled"`; rejects unknown arms; rejects
`denoiser_arch="shared_recursive"`); bit-exact-when-disabled parameters and
training loss; the connector never participates in the training-loss
gradient graph; construction wires `denoiser._plan_connector` correctly;
`generate_with_plan_connector` raises when the connector is disabled.

`generate_with_plan_connector` itself was not exercised end-to-end by a
generation call in this sandboxed development environment: even an unrelated
baseline `TwoTowerModel.generate(...)` call (no AP-024 code involved) did not
return within a multi-minute budget here after a short (20-step) training
loop, matching other pre-existing slow/hanging generation paths observed
during this session's harness work -- a sandbox characteristic, not a defect
introduced by this change (confirmed by reproducing the identical hang on the
unmodified merge-base commit). Every underlying primitive the method composes
(`abstract_plan_trace`, `plan_vector_from_trace`, `resolve_plan_vector`,
`set_plan_vector`, `project`'s bias application, `pop_plan_connector_traces`)
is covered directly and does return fast in this environment.
