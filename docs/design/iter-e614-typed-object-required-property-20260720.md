# E614 — typed-object required-property closure

Date: 2026-07-20
Status: completed, wiring/fixture evidence only (no checkpoint available)

E613 floored the schema-derived object opener for authored typed-array items
(`ImageGallery.images`) but the resulting object frame did not retain the
array item's schema, so decoding filled arbitrary keys and nested components
out to the 160-token canvas. E613's own next-step note asked to "propagate
typed object property schemas into the choice state, require known keys, and
make required property closure explicit before testing another decode
margin."

E614 does exactly that at the grammar layer:

- `ChoiceDecodeState` (`src/slm_training/models/choice_tokenizer.py`) gains an
  opt-in `require_object_schema_properties` flag. When set, opening `{` inside
  a typed-array item (or typed component argument) whose schema is
  `type: object` now records that schema's `required` property names on the
  new `_ChoiceFrame` (`object_required`) and tracks which have been filled
  (`object_filled`) as `n:<name>` key tokens are consumed.
- Closing the object (`}`) is grammar-rejected (`advance_id` returns `False`)
  while any required property remains unfilled, so `allowed_ids` no longer
  offers `}` early and offers the missing `n:<name>` token instead.
- `_completion_id` (used by `minimal_completion_length`, which gates
  `allowed_ids`' length-feasibility check) was updated to target the next
  missing required property's minimal `n:<name>` token before falling back to
  `}`, so the new constraint does not turn every candidate at that frame into
  a false "length infeasible" rejection.
- `ChoiceDecodeState.signature()` now includes `object_required`/
  `object_filled` so the completion/allowed-id caches key correctly on this
  new state.
- The flag threads through `TwoTowerConfig.semantic_plan_typed_object_required_property_closure`
  (default `False`), `ModelBuildConfig`, `apply_runtime_overrides`,
  `_effective_evaluation_policy`, and a new
  `--semantic-plan-typed-object-required-property-closure` CLI flag on
  `scripts/evaluate_model.py`.

No prior checkpoint (`e569-e561-matched-cont48-r1-48s`, reused by E608-E613)
is present under `outputs/` in this environment, so a matched OOD
quality-metric replay comparable to E608-E613's aggregate numbers could not be
run this iteration. Instead,
`scripts/run_e614_typed_object_required_property_fixture.py` exercises the
grammar directly against the real `ImageGallery` component contract
(`images[].src` required):

- Default off: `+ImageGallery([{` may legally close with zero keys
  (reproduces E613's failure-mode precondition).
- Opt-in on: `}` is no longer legal until `n:src` is filled;
  `minimal_completion_length` stays finite (4 default-off legal tokens to
  `src=""` well-formedness, 6 opt-in — the extra key/value pair — never
  `>=1025`, i.e. never "infeasible").
- Filling `n:src` then a value permits a full legal completion to `eos`.

All 5 checks pass. This is grammar-level wiring/fixture evidence, not a
quality-metric claim: strict meaning-v2, AgentV, and the E608-E613 aggregate
scoreboard are untouched by this iteration, and the lever remains default off.
No checkpoint was created, promoted, or synced.

Next iteration: once a checkpoint (e.g. a freshly bootstrapped or synced
`e569-e561`-equivalent) is available, run the matched OOD `n=4` replay with
`--semantic-plan-typed-object-required-property-closure` stacked on E611's
retained levers to measure whether Gallery's forced `src` key repairs E613's
structure/latency regression instead of just its empty-array/arbitrary-key
failure modes.

Evidence: [JSON](iter-e614-typed-object-required-property-20260720.json).
