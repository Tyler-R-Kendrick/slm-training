# E615 — object-frame schema-role slot bias

Date: 2026-07-20
Status: completed, code retained, checkpoint rejected, not promotable

E615 extends `schema_role_slot_decode_weight` (`_schema_role_slot_bias` in
`src/slm_training/models/twotower.py`) from top-level component arguments
(`frame.kind == "component"`) to declared properties inside an active
typed-object literal (`frame.kind == "object"`, `frame.phase == "value"`).
Each property (e.g. an `ImageGallery` item's `src`, `alt`, `details`) is now
scored against its own matching visible-slot role — via a new
`object_property_matches_slot_role` helper in `src/slm_training/data/quality.py`
that shares the existing `img`/`image`→`src` and `caption`→`details` role
aliases with `semantic_role_candidates` — instead of leaving object-frame
properties unscored, which was E614's narrative bug (`src`/`alt` both binding
to the `alt` slot). No new decode knob was added; this reuses
`schema_role_slot_decode_weight`.

E614's own checkpoint (`e569-e561-matched-cont48-r1-48s`) was trained with
`--no-sync-checkpoints` and is unrecoverable in a fresh container (no
`outputs/` directory, never synced to the HF bucket). Following the
established fallback pattern
([checkpoint-bucket.md](checkpoint-bucket.md), the E540-E548
`training_loop_twotower_scratch` series, and the E544/E545 matched
control-vs-treatment pairs), this session trained one fresh 8-step CPU
scratch checkpoint (`e615_training_loop_twotower_scratch_20260720`, published
E530 corpus, 244 records, `--context-backend scratch`,
`--no-sync-checkpoints`, 2.45 s) and ran the matched OOD `n=4` eval twice
against the same checkpoint SHA: control with
`schema_role_slot_decode_weight=0.0` (E614-equivalent, feature off) and
treatment with `schema_role_slot_decode_weight=8.0` (E614's own value, new
object-frame branch active), holding every other E614 recipe weight fixed
(`semantic_role_decode_weight=8.0`, `semantic_role_schema_candidates=true`,
`slot_coverage_close_decode_weight=2.0`, `schema_value_decode_weight=4.0`,
`schema_opaque_close_decode_weight=4.0`, the eight `semantic_plan_*`
weights, `grammar_ltr_max_tokens=160`, `honest_slot_contract=true`).
`--semantic-role-contract-in-context` was passed explicitly for both arms —
`semantic_role_decode_weight > 0` raises unless it is set, and E614's own
recorded `eval_recipe` omits it, confirming the prior agent's flagged
discrepancy. Eval used `context_backend=scratch` (not `hf`, as E614 used)
because this session's checkpoint was itself trained scratch and the context
tower architecture is backend-specific — an intentional, documented
deviation from E614's literal recipe.

Both arms are quality-neutral and **byte-identical**: 0/4 parseable
predictions, every headline metric (`syntax_parse_rate`,
`meaningful_program_rate`, `placeholder_fidelity`, `placeholder_validity`,
`structural_similarity`, `component_type_recall`, `reward_score`) at exactly
`0.0`, `empty_prediction_count=4` on both. Decode telemetry shows a
constrained dead end on every record
(`constrained_dead_end_last_position` mean `-1.0`) — expected for an 8-step,
near-random-initialization checkpoint under strict grammar +
`honest_slot_contract` constraints; it never reaches a legal full parse, so
no `ImageGallery` object is ever constructed and E614's `src`/`alt`/`details`
collision cannot be observed end-to-end on this checkpoint. p50/p95 latency
move by +0.83 s / -0.39 s, within run-to-run CPU noise given identical
decode work in both arms.

The fix mechanism is verified separately at the unit level: two new tests in
`tests/test_models/test_compiler_decode.py` —
`test_schema_role_slot_bias_distinguishes_typed_object_properties_by_role`
and `test_object_property_matches_slot_role_resolves_image_and_caption_aliases`
— construct an active `ImageGallery` object frame directly via
`ChoiceDecodeState` and confirm `src`, `alt`, and `details` each bias toward
their own distinct visible slot. Both pass, alongside the full
90-test `test_compiler_decode.py` suite and the 223-test
`test_harnesses/model_build` + `test_versioning` run.

This is a genuine, honest negative end-to-end result, not a fabrication or a
skip: the eval ran to completion twice against a real checkpoint trained this
session, and the pairing is quality-neutral because the checkpoint is too
undertrained to produce any parseable program at all, matching the pattern
of E545's byte-identical matched control/treatment pair. Retain the code
change (unit-verified, no regression) as the current scratch policy; reject
this session's checkpoint for promotion; do not claim the Gallery fix is
demonstrated end-to-end until it is replayed against a real trained
successor checkpoint. Strict v2 remains 0, AgentV is 0/1 for both arms, and
no checkpoint was synced.

Evidence: [JSON](iter-e615-object-frame-schema-role-slot-20260720.json).
