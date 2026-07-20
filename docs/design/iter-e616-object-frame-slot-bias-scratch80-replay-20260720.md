# E616 — object-frame slot-bias replay on an 80-step scratch checkpoint

Date: 2026-07-20
Status: completed, code retained (unchanged this iteration), checkpoint rejected, not promotable

E615 added an object-frame branch to `schema_role_slot_decode_weight`
(`_schema_role_slot_bias` in `src/slm_training/models/twotower.py`) so a
typed-object property being filled during grammar-guided decode (e.g. an
`ImageGallery` item's `src`, `alt`, `details`) scores against its own matching
visible-slot role instead of leaving object-frame properties unscored. E615's
own matched OOD `n=4` control-vs-treatment eval was byte-identical because its
8-step scratch checkpoint (loss 37.9045) never emitted a non-empty prediction
on any of the 4 OOD records — the checkpoint was simply too undertrained to
complete constrained decode at all. E615's own "next" note called for
replaying the same matched pair "when a sufficiently trained successor
checkpoint exists."

This iteration is that replay attempt. No prior E608-E614 lineage checkpoint
(`e569-e561-matched-cont48-r1-48s`) is recoverable in this fresh container
(empty `outputs/`, never synced to the HF bucket), so — following the
established E540-E548/E615 fallback pattern — a fresh scratch checkpoint was
trained again from the same published corpus, this time for 80 steps instead
of 8 (`e616-object-property-slot-bias-scratch80-20260720`, 244 records,
`--context-backend scratch --output-tokenizer choice --no-sync-checkpoints`,
12.78s wall, final loss 26.5243 vs E615's 37.9045). No code changed this
iteration; `python -m scripts.verify_version_stamps --check` confirms 0
components touched.

The matched OOD `n=4` eval was run twice against the identical checkpoint SHA
(`119dd41a…eef0c508`): control with `schema_role_slot_decode_weight=0.0`
(feature off) and treatment with `=8.0` (E614/E615's value), holding every
other E614/E615 recipe weight fixed (`semantic_role_decode_weight=8.0`,
`semantic_role_schema_candidates`/`semantic_role_contract_in_context=true`,
`slot_coverage_close_decode_weight=2.0`, `schema_value_decode_weight=4.0`,
`schema_opaque_close_decode_weight=4.0`, the eight `semantic_plan_*` weights,
`grammar_ltr_max_tokens=160`, `honest_slot_contract=true`,
`context_backend=scratch` to match the checkpoint's own backend).

**Progress over E615:** `syntax_parse_rate` rises from 0.0 to 1.0 — all four
OOD records now decode to syntactically valid, non-empty programs in both
arms (structural similarity 0.2246, AST-node F1 0.2637, component-type recall
0.1875). This is a genuinely different regime from E615's checkpoint, which
produced 0/4 non-empty predictions.

**Still byte-identical, but now for a diagnosable reason.** The record that
exercises the E615 lever, `ood_gallery_01`, decodes identically in both arms
to:

```openui
root = Stack([v0], "column")
v0 = ImageGallery([])
```

The typed array closes **empty** before any item object is ever opened. The
E615 object-frame branch of `_schema_role_slot_bias` only fires once a `{`
has been opened inside a typed-array item (`frame.kind == "object"`,
`frame.phase == "value"`); since that never happens on this checkpoint,
control and treatment are mechanically forced to agree regardless of the
weight — not because the lever is broken, but because its precondition is
never reached. This reproduces E612's already-documented, already-rejected
finding (an undertrained checkpoint prefers closing a typed array empty over
opening an item) rather than exposing a new defect.
`semantic_plan_typed_array_nonempty_margin_decode_weight=2.0` was active in
both arms and did not change this outcome on this checkpoint class.

Every headline quality metric (`meaningful_program_rate`,
`binding_aware_meaningful_v2_rate_strict`, `placeholder_fidelity`,
`placeholder_validity`, `reward_score`) is exactly `0.0` in both arms, and all
4 predictions are byte-identical between control and treatment (not just
aggregate metrics — verified at the per-record `prediction` string level).
AgentV is 0/1 for both arms; strict meaning-v2 remains 0.

The E615 fix mechanism remains verified only at the unit level
(`tests/test_models/test_compiler_decode.py::test_schema_role_slot_bias_distinguishes_typed_object_properties_by_role`
and `::test_object_property_matches_slot_role_resolves_image_and_caption_aliases`,
both passing, alongside the full 103-test
`test_compiler_decode.py` + `test_versioning` run this session). This is a
genuine, honest negative end-to-end result on real checkpoints and real
evals, not a fabrication or a skip: two real matched OOD eval pairs (E615's
8-step and this session's 80-step) have now been run, and both are
quality-neutral for the same underlying reason one level removed — decode
never reaches the point the E615 lever biases. Retain the E615 code change;
do not claim the Gallery `src`/`alt`/`details` fix is demonstrated
end-to-end until either a heavily-trained/warm-started successor checkpoint
is available, or the typed-array-nonempty decision itself (adjacent to, and
already partly investigated by, E612) is revisited. No checkpoint was
promoted or synced.

Evidence: [JSON](iter-e616-object-frame-slot-bias-scratch80-replay-20260720.json).
