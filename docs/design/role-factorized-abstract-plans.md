# AP-025 (SLM-318): role-factorized abstract token families and slot-swap probes

Status: harness wiring only. **No training run, no matched-arm quality
comparison, and no held-out semantic-delta measurement was executed as part
of this change** -- this document registers a data contract, a masking hook,
and a probe utility, not an experiment result. In particular this is not a
claim that role-factorizing the AP-016 (SLM-302) codebook improves anything;
that quality/necessity claim is AP-022's (SLM-313) job, over paired locked
examples and a real scorer, once this wiring exists.

## Decision

Test typed discrete plan channels (intent, inventory, cardinality, topology,
bindings, style -- the same family names `semantic_plan_factors.py` already
uses for the richer `SemanticPlanV1`) against the homogeneous AP-016
codebook, while retaining the homogeneous variant unchanged and at matched
parameter count.

## What this adds

`src/slm_training/dsl/abstract_plan.py`:

- **`RoleSpanV1`** -- a frozen, validated `(role, start, length, max_length)`
  record: one named, contiguous, non-overlapping range of codebook indices.
  `max_length` is an optional per-role budget on how many *occupied* rounds
  of that role survive a truncation probe (see below); it must be `<=
  length` when set.
- **`AbstractPlanV1.role_spans`** -- an ordered tuple of `RoleSpanV1`,
  `()` by default (homogeneous, matching AP-016's original contract exactly).
  Validated for ascending, non-overlapping ranges, unique role names, and
  ranges within `slot_count`. **Mutually exclusive with the existing
  per-slot `role_metadata`** -- a slot must not carry two conflicting role
  representations -- so a caller picks one role encoding per plan.
  `role_for_slot`, `role_slot_indices`, `role_slot_token_ids`, and
  `role_slot_mask` map between codebook indices/absolute vocabulary ids and
  role names; `is_role_factorized`/`role_names`/`role_span` round out the
  read surface.
- Crucially, **`role_spans` reserves no new vocabulary token and changes no
  token id** -- it only labels which of the *existing* `slot_count` codebook
  indices belong to which role. `token_ids`, `assert_no_collisions`, and
  `codebook_version` are all unaffected; a role-factorized plan and a
  homogeneous plan with the same `slot_count` have byte-identical
  `token_ids`.

`src/slm_training/models/abstract_plan_head.py`:

- **`AbstractPlanHead.forward(..., role_sequence=None)`** -- an optional,
  per-round list of role names (`len(role_sequence) == plan.rounds`) that
  restricts each round's legal codebook range to its declared role before
  computing entropy/sampling. This reuses the **same** `nn.Linear` projection
  used by the homogeneous variant -- masking its output, not adding a second
  projection -- so a homogeneous arm and a role-factorized arm trained with
  this hook are matched at **identical parameter count and identical
  per-round compute**, directly satisfying the acceptance criterion "compare
  variants at matched tokens, parameters, and compute" without a new
  architecture to keep in sync.
  - `None` (the default) is bit-exact identical to the pre-AP-025 forward
    path for every mode -- proven by
    `test_role_sequence_none_is_bit_exact_with_no_role_sequence_argument`.
  - `RANDOM` mode (which does not run the projection) gets its own masked
    uniform sampler, `_sample_masked_uniform`, so the "matched arms" property
    holds for the content-null control arm too, mirroring
    `abstract_plan_connector.py`'s `PlanConnectorArm.RANDOM` idiom of paying
    identical shape/device/dtype cost regardless of arm.
  - `target_plan_ids` outside the declared role fail closed with a
    `ValueError`, rather than silently accepting a gold plan that
    contradicts its own role assignment.

`src/slm_training/dsl/role_slot_probe.py` (new module):

- **`swap_role(plan, tokens_a, tokens_b, role)`** -- cross-example
  intervention: exchanges the rounds whose emitted slot belongs to `role`
  between two same-length plan-token sequences, leaving every other round
  untouched in both sequences. A single-factor counterfactual, not a
  full-plan shuffle.
- **`dropout_role(plan, tokens, role)`** -- replaces `role`'s occupied
  rounds with that role's first slot (a deterministic, content-null filler,
  the discrete-codebook analog of `PlanConnectorArm.EMPTY`).
- **`truncate_role(plan, tokens, role, keep=None)`** -- keeps only the first
  `keep` occupied rounds of `role` (defaulting to its `max_length` budget,
  or full `length` when unset), filling the rest with the same role filler.
- **`run_role_probes`/`causal_effect_matrix`** -- pure orchestration over a
  sequence of example pairs and a caller-supplied `SemanticScorer` protocol
  (the only model-dependent surface, mirroring
  `harnesses/preference/counterfactual_probe.py`'s `RolloutBackend`
  protocol), producing the raw per-`(role, intervention)` mean-absolute-delta
  matrix the acceptance criteria call a "cross-factor causal effect matrix".

`scripts/run_role_slot_probe.py` -- a bounded, fixture-only demonstration:
a 12-slot, 6-role plan, two hand-written token-sequence pairs, and a mock
"fraction distinct" scorer. Explicitly labeled a wiring demo, not a quality
result (see "Deferred, honestly" below).

## Why role_spans don't reserve new tokens

`role_metadata` (AP-016) already had a slot-labeling mechanism, but it is
flat and per-slot -- no ordered ranges, no non-overlap guarantee, no
delimiters, no length budget. The acceptance criteria ask for "role
delimiters/masks and per-role length budgets". Two designs were available:

1. Reserve new delimiter *tokens* per role inside the `abstract_plan`
   namespace (`0x4000-0x7000`), threaded through `dsl_tokenizer.py` and
   `choice_tokenizer.py`'s existing `abstract_plan_slots` build-time knob.
2. Represent roles as **metadata over the existing slot indices** -- ranges
   plus boolean masks derived from them -- with no new reserved token text.

This change takes (2). It keeps the change scoped to the data contract and
the head's masking hook, avoids touching the tokenizer append-only vocab
contract (which AP-016 already established as a careful, collision-audited
surface), and still satisfies "delimiters/masks": `role_slot_mask` *is* the
mask, and the ordered, non-overlapping `RoleSpanV1` ranges *are* the
delimiters, just expressed as boundaries over slot indices rather than as
new emitted tokens. If a later experiment finds that content-level
delimiter tokens (visible to the decoder, not just to the head/probe layer)
are actually needed, that is follow-on work through the tokenizer's
existing append-only knob -- not represented as done here.

## Deferred, honestly

- **No matched-arm training run.** "Train matched homogeneous and
  role-factorized arms under identical total token/parameter budgets" is an
  experiment, not a data-contract change. This wiring makes that experiment
  possible at truly matched parameter count (same `nn.Linear`, masked
  output), but running it -- and reporting meaning-v2/binder-reference-F1
  deltas with confidence intervals -- is AP-022's (SLM-313) job.
- **No real semantic scorer.** `scripts/run_role_slot_probe.py`'s
  `_MockScorer` counts distinct slot values; it is illustrative wiring only,
  explicitly labeled as such in the report's `note` field. A ship-grade run
  swaps in a real meaning-v2/binder-reference-F1 measurement over paired
  locked examples (`data/eval/manifests/abstract_planning_locked_v1.jsonl`),
  behind the same `SemanticScorer` protocol.
- **No decoder-side role conditioning.** This change does not touch
  `abstract_plan_connector.py` (AP-024) or the MaskGIT round loop --
  `role_sequence` only restricts what `AbstractPlanHead` predicts/samples,
  it does not change what the denoiser conditions on. Combining
  role-restricted heads with the AP-024 connector, if useful, is separate
  follow-on work.
- **No cross-example role-swap dataset.** `swap_role` operates on two
  caller-supplied plan-token sequences; building an actual paired corpus for
  the probe (as opposed to hand-written fixture pairs) is data work, not
  represented as done here.

## Acceptance criteria mapping

- "Extend AbstractPlanV1 with optional ordered role spans and non-overlapping
  per-role codebook ranges." -- `RoleSpanV1` / `AbstractPlanV1.role_spans`,
  validated ordered/non-overlapping/in-range in `__post_init__`.
- "Add role delimiters/masks and per-role length budgets while retaining the
  homogeneous variant." -- `role_slot_mask` (masks), `RoleSpanV1` range
  boundaries (delimiters, see design note above), `RoleSpanV1.max_length`
  (length budgets); `role_spans=()` is the unchanged homogeneous variant,
  proven bit-exact by `test_role_spans_do_not_change_token_ids_or_collide`
  and `test_role_sequence_none_is_bit_exact_with_no_role_sequence_argument`.
- "Train matched homogeneous and role-factorized arms under identical total
  token/parameter budgets." -- `role_sequence` reuses the single existing
  projection (zero new parameters); the actual training run is deferred to
  AP-022 (SLM-313), stated explicitly above.
- "Implement cross-example role swaps, dropout, and truncation; map
  resulting semantic deltas." -- `swap_role`/`dropout_role`/`truncate_role`
  plus `run_role_probes`/`causal_effect_matrix` in `role_slot_probe.py`.
- "Compare variants at matched tokens, parameters, and compute." -- see the
  `role_sequence` masking design above; no new parameters or projection.
- "Publish a cross-factor causal effect matrix." -- `causal_effect_matrix`;
  `scripts/run_role_slot_probe.py` publishes a fixture-only instance of it
  (explicitly not a quality claim -- see "Deferred, honestly").
- "A role intervention should primarily affect its declared factor." --
  `swap_role`/`dropout_role`/`truncate_role` only ever touch rounds whose
  emitted slot belongs to the declared `role`; every other round is passed
  through unchanged by construction (not merely by convention), so no
  intervention can leak into an undeclared factor.
- "Do not claim individual-token interpretability without consistent
  held-out causal effects." -- this document and the probe script's `note`
  field state explicitly that no causal-necessity claim is made here.

## Tests

`tests/test_dsl/test_abstract_plan.py` -- `RoleSpanV1` valid/invalid
configurations, round-trip; `AbstractPlanV1.role_spans` overlap/duplicate/
out-of-range/mutual-exclusivity-with-`role_metadata` rejection,
`is_role_factorized`/`role_for_slot`/`role_slot_indices`/
`role_slot_token_ids`/`role_slot_mask`, to_dict/from_dict round-trip, and
the token-id/collision bit-exactness-vs-homogeneous check.

`tests/test_dsl/test_role_slot_probe.py` -- `slot_roles` mapping,
`swap_role`/`dropout_role`/`truncate_role` exact-value checks (including the
"only the declared role's rounds move" property and the `max_length`-vs-
explicit-`keep` distinction), `run_role_probes` role/intervention coverage
and config restriction, `causal_effect_matrix` aggregation, and
requires-role-factorized-plan rejection for every probe function.

`tests/test_models/test_abstract_plan_head.py` -- `role_sequence=None`
bit-exactness (both against the previous no-argument call and against an
explicit `None`), requires-role-factorized-plan rejection, wrong-length
rejection, sampled- and random-mode token restriction to the declared role's
codebook range, and `target_plan_ids`-outside-declared-role rejection.
