# E632 — schema-identifier decode weight (rejected: destabilizes Auth closure)

Date: 2026-07-21
Status: completed negative; new knob kept default-off; not ship

E631 closed the component-inventory gap for the Auth OOD record
(`Stack([Button, Input, Input])` now has perfect placeholder fidelity,
validity, structural similarity, and component recall) but strict meaning-v2
still fails there: both sibling `Input`s duplicate the `name`/`email` slot
values across `Input.name` *and* `Input.placeholder`, and the evaluator flags
the `Input.name` occurrences as semantic-role mismatches. E631's explicit
next step was to "resolve Input property-role assignment against the
gold/evaluator contract ... prefer literal control names/types and place
visible form-field content slots in the schema property justified by the
authored prompt." E632 tests the most direct fix: a new decode-time bias that
discourages visible-slot placeholders from landing in required, non-enum,
non-content string identifier properties (schema type `"string"`, not tagged
`x-openui-placeholder`, e.g. `Input.name`), so the model is pushed toward a
literal token there (as gold does: `Input("text", ":slot")`) and the slot
value only fills the real content property (`Input.placeholder`).

## Code change

`src/slm_training/models/twotower.py` gains `_schema_identifier_bias`
(mirrors the existing `_schema_value_bias`/`_schema_opaque_bias` family,
gated by a new default-off `schema_identifier_decode_weight` config field)
and is wired into the same decode-loop bias stack right after
`_schema_value_bias`. `tests/test_models/test_compiler_decode.py` gains
`test_schema_identifier_bias_penalizes_slots_only_for_non_content_identifiers`,
which confirms in isolation that the bias penalizes `Input.name` (plain
string, non-enum, non-content) and stops penalizing once the frame advances
to `Input.placeholder` (content-tagged). The new weight is plumbed through
`config.py`, `factory.py` (`apply_runtime_overrides` + build-time
construction), `eval_runner.py`, and a new `--schema-identifier-decode-weight`
CLI flag on `scripts/evaluate_model.py`, following the exact pattern of
`schema_role_slot_decode_weight`.

## Retrained checkpoint and recipe

E620/E631's local-only checkpoint (sha256 `3ce5c9ef...`) was not present in
this session's sandbox (`outputs/` is not persisted across sessions), so this
iteration retrained E620's identical 800-step scratch recipe instead of
skipping the eval: same train dir (`e530_visible_semantic_roles_r2_20260719`,
244 records), model, scratch context, choice output tokenizer, device cpu,
batch size 1, seed 0, `--no-sync-checkpoints`. It completed in 32.2 seconds.
The resulting checkpoint's SHA-256 is
`ce32c228ef62bf4d15176e338102b0e4a18f2cebb7e012ea0e4efec3931ff0c4` (new, since
it is a fresh train, not the same file) but its **control-arm predictions on
this OOD `n=4` suite are byte-identical to E631's treatment-arm predictions**
on all 4 records, confirming this is a faithful replay of the same lineage
before adding this iteration's new bias. It is local-only, not synced, not
promoted.

Both eval arms used the identical checkpoint and E631's exact policy
(`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, `semantic_role_contract_in_context`, public
semantic-role schema candidates, and the full retained semantic-plan/closure
weight stack). Only `schema_identifier_decode_weight` changed: 0 for control,
4.0 for treatment (matching `schema_value_decode_weight`'s magnitude). Both
runs emitted AgentEvals JSONL and an AgentV SDK bundle with no execution
errors and completed well under the three-minute cap (each eval command ran
in well under 30 seconds).

## Measured result

| OOD `n=4` | Control (weight=0) | Treatment (weight=4.0) |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 |
| v2 judgment coverage | 1.0000 | 1.0000 |
| placeholder fidelity | 0.6750 | 0.5917 |
| placeholder validity | 0.8050 | 0.7550 |
| structural similarity | 0.5729 | 0.4704 |
| component recall | 0.6250 | 0.5000 |
| reward | 0.8515 | 0.8205 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.5524 / 0.3875 |
| latency p50 / p95 | 1275.16 / 5073.26 ms | 1606.58 / 5086.41 ms |
| closure applications / changes | 25 / 12 | 22 / 9 |
| AgentV | 0/1 | 0/1 |

Every continuous quality metric regresses. Dashboard, Gallery, and Modal
predictions are byte-identical between arms (their property structure never
traverses a `schema_identifier_decode_weight`-guarded arg position). The
entire regression is the Auth record: the treatment prediction is
`root = Stack([v0, ":ood.auth.create", ":ood.auth.email", ":ood.auth.create", []])`
— the model abandons the `Input(...)` wrapper entirely and emits bare slot
references as direct `Stack` array items, plus a stray empty array, instead
of E631's `Stack([Button, Input, Input])`.

Three additional single-flag probes (0.25, 0.5, 1.0), holding every other
weight fixed, show a knife-edge: weight 0.25 is byte-identical to the control
(no effect at all), while 0.5 and 1.0 are already byte-identical to the 4.0
treatment (fully destructive). There is no intermediate magnitude in this
session that fixes the `Input.name`/`Input.placeholder` duplication without
destabilizing the Auth closure.

## Analysis

`_schema_identifier_bias` behaves exactly as the new unit test verifies in
isolation: it only discourages a visible-slot candidate at a "component"
frame's arg position when that position's schema is a required, non-enum,
plain `"string"` type not tagged `x-openui-placeholder` — precisely
`Input.name` and not `Input.placeholder`. But this checkpoint's decoder is a
parallel MaskGIT-style denoiser, not autoregressive, so a bias that looks
locally scoped to one arg position can still change the *global* decode
trajectory: once the model's only confident continuation at `Input.name` is
penalized, confidence-ordered remasking changes which positions get resolved
first and in what order elsewhere in the canvas.
`slot_coverage_close_applications` drops 25 -> 22 (`choice_changes` 12 -> 9)
and `semantic_plan_root_applications` drops 16 -> 7 (decode-stats sums, not
shown in the headline table) — concrete evidence that the intervention's
effect is not contained to the single arg position it targets. On this
800-step scratch checkpoint, the model apparently never learned a strong
independent preference for a literal token at `Input.name`; once its only
confident option there is suppressed, the higher-level "open two sibling
`Input`s" decision itself becomes unstable and the model retreats to a
simpler (but wrong) legal completion: bare slot references directly as array
items.

This does not falsify the underlying idea (gold really does put a literal
control token in `Input.name` and the placeholder in `Input.placeholder`,
and the schema really does distinguish the two by `x-openui-placeholder`) —
it falsifies this specific decode-time-only enforcement mechanism *on this
specific undertrained checkpoint*. A better-trained checkpoint that already
has a genuine preference for literal identifiers might tolerate the bias
without needing to destabilize the array-closure decision; this session's CPU
budget does not support training to that point.

## Decision

Reject `schema_identifier_decode_weight` as a decode-time treatment at every
magnitude tested. Keep the new bias function, config field, and CLI plumbing
in the codebase as a default-off (`weight=0.0`) knob — it is unit-tested in
isolation, changes no existing recipe's behavior while left at 0.0, and
remains available for a future, better-trained checkpoint or a decode-order
-aware follow-up (e.g. gating the bias on a minimum decode-position
confidence, or applying it only after the array-closure decision has already
committed). No recipe in this repository enables it. Do not sync, promote, or
claim ship readiness. The Input property-role assignment problem E631 left
open remains open; a future iteration should look at training-time signal
(e.g. an auxiliary loss that teaches literal `Input.name` values directly)
rather than a purely decode-time penalty, given this checkpoint's MaskGIT
instability under position-local biases.

Evidence: [JSON](iter-e632-schema-identifier-decode-weight-20260721.json).
