# E620 — the E617 bug class had a second, still-live instance

Date: 2026-07-20
Status: completed, real bug found and fixed (masked-not-live in current
recipes), real before/after confirmation on the real E617 checkpoint, no
checkpoint trained/promoted/synced

E619 closed the `--slot-contract-in-context` question left open by E618. This
iteration's brief was to broaden rather than repeat: systematically audit
`twotower.py`/`choice_tokenizer.py` for more instances of E617's failure
class (a decode-time bias reading per-row state that's populated only when an
orthogonal, easily-forgotten flag is set) and audit `meaningful_program.py`
and its siblings for more instances of E618's failure class (a
regex/heuristic fallback misclassifying syntactically-valid current-grammar
output as an error). This finds and fixes a real, live-shaped instance of the
first class.

## Method

Rather than re-read the whole file end to end, this session traced every
`self._*` attribute in `TwoTowerModel` that is (a) populated conditionally
during batch/decode setup and (b) read by a decode-time bias function that
silently returns `None`/no-ops when that state is absent. Four such
attributes exist: `self._slot_contracts`, `self._semantic_role_candidates`,
`self._semantic_plan_action_scores`, `self._semantic_plan_action_counts`
(`twotower.py`, declared ~line 948). `choice_tokenizer.py` has no analogous
per-row gated state at all (grepped for `self\._[a-zA-Z_]+\s*=`: no hits) —
clean, ruled out for this bug class.

- `self._slot_contracts`: already fixed by E617. Traced every reader
  (`_schema_role_slot_bias`, `_slot_coverage_close_bias`,
  `_semantic_plan_typed_array_nonempty_bias`,
  `_semantic_plan_repeated_slot_bias`, plus the inline
  `pick_constrained_token` call sites) against E617's `ValueError` guard's
  5-name list (lines 8082-8114 pre-fix). All 5 covered; no 6th reader found.
  Clean.
- `self._semantic_role_candidates`: self-guarding — `semantic_role_decode_weight
  > 0` already raises `ValueError` if `honest_slot_contract` or
  `semantic_role_contract_in_context` isn't also set (lines 8134-8140). Clean.
- `self._semantic_plan_action_scores`/`self._semantic_plan_action_counts`:
  populated whenever `plan_weight = max(...)` over a **hand-enumerated
  9-name list** (pre-fix lines 8161-8196) is positive. **This is the finding.**

## Root cause

`TwoTowerConfig` declares 11 fields matching the
`semantic_plan_*_decode_weight` / `semantic_plan_*_margin_decode_weight`
pattern (verified via `grep` against the dataclass body). The pre-fix
`plan_weight` computation enumerated only 9 of them. The two missing names —
`semantic_plan_inline_decode_weight` and
`semantic_plan_repeated_array_close_margin_decode_weight` — are each the sole
gate for a real decode-time bias that reads exactly this same state:

- `_semantic_plan_inline_bias` (`twotower.py:4729`) reads
  `self._semantic_plan_action_scores`/`_action_counts` at lines 4742-4749,
  gated on `semantic_plan_inline_decode_weight`.
- `_semantic_plan_repeated_array_close_bias` (`twotower.py:5141`) reads
  `self._semantic_plan_action_counts` at line 5161 (directly, and again via
  `_semantic_plan_repeated_owner_id`), gated on
  `semantic_plan_repeated_array_close_margin_decode_weight`.

If either of those two weights were the *only* nonzero `semantic_plan_*`
weight in a recipe, `plan_weight` would evaluate to `0.0`, the `if
plan_weight > 0.0:` branch would be skipped, and
`self._semantic_plan_action_scores`/`_action_counts` would stay `None` for
the whole batch — both biases would silently no-op on every decode step,
independent of weight value or checkpoint quality. This is the exact same
failure shape as E617's `self._slot_contracts` gap, one level removed: not a
missing companion *flag*, but a hand-maintained *list* of weight names that
drifted out of sync with the functions that actually consume the state it
gates.

**Currently masked, not live-broken.** Grepping every `evaluation_policy` /
`eval_recipe` block in `docs/design/iter-e610*.json` through
`iter-e619*.json` confirms `semantic_plan_decode_weight` (which *is* in the
9-name list) is set to a nonzero value (4.0) in every one of them, alongside
`semantic_plan_repeated_array_close_margin_decode_weight=2.0`. That
coincidence has kept `plan_weight` positive and the state populated in every
real E610-E619 eval to date — so this gap has not silently corrupted any
already-reported result in this lineage. But `semantic_plan_inline_decode_weight`
has never been set nonzero in any committed recipe (dormant, same landmine,
zero exercised risk so far), and any future isolated ablation of
`semantic_plan_repeated_array_close_margin_decode_weight` alone — exactly the
style of matched control/treatment diagnostic this lineage keeps running —
would have hit this silently, the same way E611-E616 silently hit the
`self._slot_contracts` gap for six experiments before E617 found it.

## Fix

Replaced the 9-item hand-enumerated `max(...)` literal in
`_generate_batch_once` with `max(getattr(self.config, name, 0.0) or 0.0 for
name in SEMANTIC_PLAN_DECODE_WEIGHT_NAMES)`, where
`SEMANTIC_PLAN_DECODE_WEIGHT_NAMES` is a new module-level tuple
(`twotower.py`, defined immediately after `TwoTowerConfig`) listing all 11
weight names, including the two previously-missing ones.

**Regression coverage** (`tests/test_models/test_compiler_decode.py`):

1. `test_semantic_plan_decode_weight_names_cover_all_config_fields` — a
   drift guard using `dataclasses.fields(TwoTowerConfig)` to assert every
   `semantic_plan_*_decode_weight`/`semantic_plan_*_margin_decode_weight`
   config field is present in `SEMANTIC_PLAN_DECODE_WEIGHT_NAMES`. This fails
   the build the next time a new `semantic_plan_*` weight field is added
   without being added to the list — the exact class of drift that caused
   this bug — instead of silently repeating it a third time.
2. `test_semantic_plan_only_weight_still_populates_plan_state`
   (parametrized over both previously-missing weight names) — calls
   `model.generate_batch_requests(...)` with only that one weight set (8.0)
   and asserts `self._semantic_plan_action_scores`/`_action_counts` end up
   populated (not `None`). This is a real, non-mocked, end-to-end generation
   call through a real (tiny) `TwoTowerModel`, mirroring E617's own test
   style.

`pytest tests/test_models/test_compiler_decode.py`: 100 passed (97
pre-existing + 3 new), `NODE_OPTIONS` unset per the documented fix.

## Real before/after confirmation on the real E617 checkpoint

No retraining. Reused the real, on-disk E617 checkpoint
(`outputs/runs/e617-debug-repro-scratch80-20260720/checkpoints/last.pt`,
sha256 `119dd41a…8898a853`, re-verified via `sha256sum` before use — matches
E617/E619's own recorded value byte-for-byte). Loaded it directly via
`TwoTowerModel.from_checkpoint`, applied E617's exact real
`evaluation_policy` block with every `semantic_plan_*` weight zeroed except
one target weight set to `8.0`, and called `generate_batch_requests` on the
real `ood_gallery_01` OOD record (`Image gallery block with caption text
underneath.`), before the fix (`git stash` of only `twotower.py`) and after.

| Isolated weight | Pre-fix `_semantic_plan_action_scores`/`_action_counts` | Post-fix |
| --- | --- | --- |
| `semantic_plan_repeated_array_close_margin_decode_weight=8.0` (all other `semantic_plan_*`=0) | `None`/`None` (bug reproduced) | populated/populated |
| `semantic_plan_inline_decode_weight=8.0` (all other `semantic_plan_*`=0) | `None`/`None` (bug reproduced) | populated/populated |

The raw prediction text on this specific 80-step scratch checkpoint and
record is **byte-identical before and after the fix in both cases** — the
state now populates correctly, but this particular garbage-quality
prediction (the same malformed, deeply-nested `Stack`/`ImageGallery` tree
E618 characterized as decode-level instability, not an evaluator artifact)
doesn't visibly change shape from this one bias alone. This is an honest,
checkpoint-quality-dominated null result at the prediction level, consistent
with — not contradicting — the state-population fix: the root cause (the
bias never even ran) is confirmed and repaired; whether a well-trained
checkpoint's output would visibly change under this bias is a separate,
unanswered question (same caveat E617/E619 already carry for their own
findings on this checkpoint class).

## Class-B audit (evaluator false positives)

Traced every check in `binding_aware_meaningful_v2()`
(`src/slm_training/evals/meaningful_program.py`) for a regex/heuristic
fallback shaped like E618's bug (misclassifying syntactically-valid,
parser-accepted output as an error):

- `binding_correctness`/`_binding_check`: the E618 fix (unconditional
  AST-based reachability walk) is in place and confirmed clean.
  `duplicate_binding` still uses a regex (`_ASSIGNMENT_RE`,
  `^\s*(\$?[A-Za-z_]\w*)\s*=`) over raw source — checked whether current
  object-frame/typed-array syntax could trip it: object-literal properties
  use `key: value` (colon), never `key = value`, confirmed against
  `docs/design/iter-e614..e619*.md`'s own examples and
  `tests/test_evals/test_meaningful_program.py`'s `$label = ":cta.label"`
  state-declaration fixture (the only legitimate `=`-at-line-start
  construct). No live false-positive path found; flagged as a latent-only
  risk, not fixed.
- `required_inventory_coverage`/`placeholder_semantic_role_mismatch`
  (`_inventory_check`): a hardcoded term/owner vocabulary
  (`_ACTION_SLOT_TERMS`/`_TEXT_SLOT_TERMS`/`_FORM_SLOT_TERMS`) applied to
  AST-derived `component`/`property` fields, not a raw-source regex — a
  different, lower-confidence risk category (missing vocabulary coverage,
  not parser misreading) than E618's bug shape. No repro constructed.
- `schema_value_role_correctness` (`_schema_role_check` →
  `data/quality.py:_schema_semantic_reasons`): fully AST-based, recurses
  through `Arr`/`Obj` node kinds via `_matches_schema_value`. Hand-traced a
  typed-array-of-objects value through it; recurses correctly. Clean.
- `anti_gaming` (`_gaming_check`): fully AST-based (`_subtree_hashes` walks
  `program.root`, placeholder counts come from the AST-walked
  `_placeholder_inventory`). No regex DSL-structure fallback. Clean.
- `_prompt_contract`'s `_INVENTORY_SECTION_RE` regex operates on the
  **prompt text**, not the model's DSL output — a structurally different bug
  class from E618 (which was specifically about misparsing valid *generated*
  DSL). Not verified further; out of this iteration's scope.
- `src/slm_training/data/quality.py`'s `semantic_contract_for_openui` (not
  in `evals/`, so outside the literal scope named in this iteration's brief,
  but noted for the next iteration) reimplements a regex-based mini-parser
  (`_ASSIGNMENT_RE`, `_DECLARATION_COMPONENT_RE`, `_IDENTIFIER_RE`,
  `_QUOTED_RE`) instead of using `dsl.parser.parse`, the same *shape* as
  E618's bug. It only fires for records carrying a legacy
  `record.meta["semantic_contract"]` field (a different, older admission-gate
  track than the `binding_aware_meaningful_v2` path E610-E620 exercise). Not
  verified whether any current builder still populates that field with
  E613+-era typed-array/nested-object syntax; flagged as a concrete,
  scoped-out follow-up rather than investigated further this iteration.

No new Class-B live bug confirmed this iteration.

## Decision

Keep the `SEMANTIC_PLAN_DECODE_WEIGHT_NAMES` fix and both regression tests —
this closes a real, evidence-backed instance of E617's bug class before it
bit a future isolated ablation of either weight, exactly the way E617 itself
was found (a matched control/treatment eval that isolates one lever). No
checkpoint trained, promoted, or synced this iteration. Not a ship claim; no
existing reported metric in this lineage changes (the gap was masked, not
live, in every prior real recipe).

## Next

1. `semantic_contract_for_openui` (`data/quality.py`) is the strongest
   remaining Class-B candidate: confirm whether any current train-data
   builder still populates `record.meta["semantic_contract"]` with
   E613+-era typed-array/nested-object syntax, and if so, whether its
   regex mini-parser misclassifies it the way the old `Gate.REFERENCES`
   fallback did.
2. `duplicate_binding`'s `_ASSIGNMENT_RE` is unconfirmed-but-clean; worth a
   direct fuzz/property test against the grammar's actual serialization
   output rather than hand-reasoning, if this evaluator sees further changes.
3. Re-run the E611/E614/E615-style isolated-lever matched-pair eval for
   `semantic_plan_repeated_array_close_margin_decode_weight` alone (now that
   the fix makes it a real, non-masked lever) on a more-trained checkpoint to
   see whether it changes real predictions once the state genuinely feeds it.

Evidence: [JSON](iter-e620-semantic-plan-weight-gate-gap-20260720.json).
