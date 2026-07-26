# AbstractPlanHead — default-off discrete plan head over context-tower states (AP-023 / SLM-315)

`slm_training.models.abstract_plan_head.AbstractPlanHead` predicts a short
sequence of `AbstractPlanV1` (AP-016 / SLM-302) codebook indices from the
`TwoTowerModel` context tower's pooled hidden state. It is a pure side
channel: `TwoTowerModel.training_loss()`/`forward()` never call it, so no
plan signal reaches the decoder unless a caller explicitly invokes the new
`TwoTowerModel.abstract_plan_trace(...)` method. Wiring an actual connector
into the decoder is AP-027's job (`TwoTowerConfig.semantic_connector`), not
this issue's.

## Modes

`AbstractPlanMode` (`disabled | teacher_forced | sampled | oracle | random |
shuffled`), stored as `TwoTowerConfig.abstract_plan_mode: str = "disabled"`:

* `disabled` (default): `TwoTowerModel.__init__` never constructs
  `AbstractPlanHead` or `AbstractPlanV1` at all — zero parameters, and
  `abstract_plan_trace()` returns `None`.
* `teacher_forced`: runs the learned projection (for its logits/entropy) but
  emits the caller-supplied `target_plan_ids` verbatim as `plan_tokens`.
* `sampled`: samples `plan_tokens` from the learned projection's categorical
  distribution.
* `oracle`: a pure bypass — never runs the projection; `plan_tokens` is the
  caller-supplied `target_plan_ids` and `logits`/`entropy` are `None`.
* `random`: a negative control — `plan_tokens` are uniform random codebook
  indices, independent of context content (`logits`/`entropy` are `None`).
* `shuffled`: another negative control — samples a plan from the learned
  projection, then randomly permutes its round order.

`random` and `shuffled` follow this repository's established convention of
shipping negative controls alongside a learned mechanism (compare the
`shuffled_difficulty`/`inverse_difficulty` arms already specified for DCA0-03,
or DSH3-27's permutation augmentation) rather than a repo-specific
`AbstractPlanHead` precedent, since none existed before this issue.

## Bit-exact feature-off guarantee

Construction is gated exactly like every other optional aux head in
`twotower.py` (`binder_arity_head`, `component_plan_head`, etc.): built only
when the mode flag requires it, and wrapped in the existing
`isolated_aux_init` helper so building it never perturbs the shared training
RNG stream. `abstract_plan_head.` is registered in
`optimizer_parameter_groups()`'s auxiliary-prefix tuple so its parameters
(when present) get their own optimizer group, matching every other aux head.

Because `abstract_plan_mode="disabled"` builds nothing, `compatibility_fingerprint()`
(which hashes `state_dict()` parameter shapes) is automatically unchanged,
and `training_loss()`/`forward()` are byte-for-byte untouched by this change
— verified by
`tests/test_harnesses/model_build/test_twotower.py::test_abstract_plan_head_is_bit_exact_with_baseline_when_disabled`
(parameter-for-parameter and loss-value equality between a model built with
no `abstract_plan_mode` kwarg at all and one with it explicitly set to
`"disabled"`) and
`::test_abstract_plan_head_never_participates_in_training_loss_graph` (with
the mode enabled, `abstract_plan_head.proj.weight.grad` is `None` after
`training_loss(...).backward()`, and every shared parameter's gradient is
still bit-identical to the fully-disabled baseline).

## Reproduction

```bash
pytest -q tests/test_models/test_abstract_plan_head.py
pytest -q tests/test_harnesses/model_build/test_twotower.py -k "abstract_plan or optional_heads_do_not_shift"
```

Both are plain unit tests, complete in seconds, well inside the repository's
hard run cap (AGENTS.md § "Hard run cap").
