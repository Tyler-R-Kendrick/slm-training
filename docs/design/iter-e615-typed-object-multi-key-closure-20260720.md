# E615 — typed-object required-property closure: broadened coverage

Date: 2026-07-20
Status: completed, wiring/fixture + unit-test evidence only (no checkpoint available)

E614 propagated a typed-array item's object schema (`ImageGallery.images[].src`)
into `ChoiceDecodeState`'s pushdown state and grammar-rejected closing the
object before its schema's required properties were filled
(`require_object_schema_properties`, default off). E614's own fixture and unit
tests, however, only drove one code path through the `OBJ_OPEN` handler in
`src/slm_training/models/choice_tokenizer.py` — the `parent.kind == "variadic"`
branch (a typed-array item) — with exactly one required key (`src`). No prior
evidence exercised the sibling `parent.kind == "component"` branch (a
directly-typed, non-array object *argument*), objects with *no* required
properties (true of every shipped directly-typed object argument today, e.g.
`Input.rules`), or more than one required key.

This iteration is environment-unchanged from E614 in one respect: no prior
checkpoint (e.g. `e569-e561-matched-cont48-r1-48s`, reused by E608-E613) is
present under `outputs/` in this environment, so a matched OOD quality-metric
replay remains deferred. Instead, following E614's own suggestion to broaden
its fixture coverage, this iteration adds three grammar-only checks — a new
fixture script (`scripts/run_e615_typed_object_multi_key_closure_fixture.py`)
and matching pytest cases in `tests/test_models/test_choice_tokenizer.py` — no
changes to `choice_tokenizer.py` itself:

1. **Component-argument branch.** A synthetic `_ChoiceFrame` pushed directly
   onto the pushdown stack (bypassing `_component_contracts()`, since no
   shipped component schema currently has a required, directly-typed object
   argument) drives the `parent.kind == "component"` `OBJ_OPEN` branch. It
   derives `object_required` from the synthetic schema exactly as the
   array-item branch does, blocks a bare `}` close, and offers the required
   key name.
2. **No-required regression.** The same synthetic-frame technique with an
   empty `required` list (matching every real directly-typed object argument
   today) confirms the opt-in flag leaves those objects unaffected — a bare
   close stays legal.
3. **Two required keys, both orders.** `ImageGallery`'s real item schema
   (`required: [src]`) is synthetically widened to `("src", "alt")` after
   opening (no shipped schema currently requires two keys). Filling `src`
   then `alt`, or `alt` then `src`, both stay blocked until the second key is
   filled and both reach a legal `}` close afterward.

**Honest finding, not a regression:** `_completion_id` (which feeds
`minimal_completion_length`'s feasibility check) always targets the first
still-missing key in the schema's *declared* `required` order, not whichever
key the caller is actually about to fill next. Filling `alt` first still
reports `n:src` as the minimal-completion target at both pre-fill steps in the
reverse-order probe. This is correct for the heuristic's actual job — proving
*a* legal, feasible completion exists, not predicting the next real token —
but is worth flagging because it means the completion-length estimate is
order-oblivious even though `advance_id`/`allowed_ids` correctly accept either
fill order. All 10 fixture checks pass; all 3 new unit tests (plus the 2
pre-existing E614 tests) pass.

Strict meaning-v2, AgentV, and the E608-E613 aggregate scoreboard are untouched
by this iteration; `require_object_schema_properties` remains default off.
No metric/gate/harness/matrix file changed
(`python -m scripts.verify_version_stamps --check` reports 0 components
touched by this change — `choice_tokenizer.py` itself was not modified, only
tests and a new, unwatched fixture script were added), so no component version
bump applies. No checkpoint was created, promoted, or synced.

Next iteration: once a checkpoint (e.g. a freshly bootstrapped or synced
`e569-e561`-equivalent) is available, run the matched OOD `n=4` replay with
`--semantic-plan-typed-object-required-property-closure` stacked on E611's
retained levers, per E614's own next-step note.

Evidence: [JSON](iter-e615-typed-object-multi-key-closure-20260720.json).

```bash
python -m scripts.run_e615_typed_object_multi_key_closure_fixture
python -m pytest tests/test_models/test_choice_tokenizer.py -k required_property_closure
```
