# E628 — excluding `frame_depth == 0` fixes `required_slot_margin_decode_weight=6`'s root hijack

Date: 2026-07-20
Status: real matched OOD replay on E626's own reused checkpoint; honest,
verified-fixed result; not a ship claim

## Which E627 mitigation this picks

E627 root-caused E626's margin=6 regression to every `required_slot_margin`
fire landing at `frame_depth == 0` (before any component/object frame opens),
where a bare visible-slot token is grammar-legal as an alternative to opening
a real component at that exact position, and named two untried mitigations:
(1) reorder the per-position bias stack so `semantic_plan_root*` runs before
`required_slot_margin`, or (2) exclude `frame_depth == 0` from
`_required_slot_margin_bias`'s target set entirely.

**This iteration implements mitigation (2).** Reordering (1) was considered
and rejected as less principled: the current stack already applies
`semantic_plan_root_margin_decode_weight`/`semantic_plan_root_decode_weight`
*after* `required_slot_margin_bias`, and at margin=2 those downstream biases
already win the race (per E627's own trace). Reordering does not remove the
race, it only changes which correction gets the last additive word on the
same score tensor — margin=6 would still be large enough to escape whichever
single downstream correction runs last, because both are just additive
deltas and the final ranking depends on relative magnitude, not application
order. The exclusion is the more principled fix because it matches what the
lever was always supposed to do: E626's own docstring frames this bias as
flooring "still-missing required slots" as argument fills, never as a
root/top-level statement choice. Removing `frame_depth == 0` from scope
leaves root/top-level choice entirely to the `semantic_plan_root*` family
(already responsible for it) with **no magnitude race at all**, at any
margin — not just at the margin values E626/E627 happened to test.

## Implementation

`TwoTowerModel._required_slot_margin_bias` (`src/slm_training/models/twotower.py`)
gained an optional `state` parameter. When `state is not None` and
`len(state.frames) == 0` (no component/object frame open yet — the exact
condition E627's `_choice_phase_evidence` already reports as `frame_depth`),
the bias now returns `None` and does not fire at all at that position,
regardless of the configured margin. The decode loop's call site
(`_generate_batch_once`) now passes `state=states[row]`. `state` is optional
specifically so lower-level direct unit-test callers that construct
candidates/scores without a full decode state keep their pre-E628 behavior
(bias always considered, no frame-depth check) — only the real decode call
site is affected.

No default changed (`required_slot_margin_decode_weight` stays `0.0`, off by
default) and the E617 `slot_contract_constrained_decode` contract-gated
`ValueError` guard is untouched. `model.twotower` v60 -> v61.

## Unit tests

New: `tests/test_models/test_compiler_decode.py::test_required_slot_margin_bias_excludes_frame_depth_zero`
— proves the bias is a no-op at `frame_depth == 0` (`state.frames == []`)
even with a genuinely-missing slot and a nonzero margin that would otherwise
floor it (the exact setup E626's own
`test_required_slot_margin_bias_floors_only_still_missing_slots` uses), still
fires unchanged at `frame_depth == 1` (an open component frame), and confirms
`state=None` preserves the pre-E628 always-fires behavior for direct callers.

Existing tests unchanged and still passing: E626's
`test_required_slot_margin_bias_floors_only_still_missing_slots` (doesn't
pass `state`, so unaffected by design) and E627's
`test_required_slot_margin_trace_flags_a_root_level_component_hijack` (tests
`_record_required_slot_margin_trace` directly with a hand-built
`margin_bias`; annotated with a note that this exact `frame_depth == 0`
hijack scenario is now unreachable via the real bias call site in
production, but the trace function's own labeling logic is still validly
covered by the test). All 133 tests pass across `test_compiler_decode.py` +
`test_decode_stats.py` + `test_choice_tokenizer.py` (includes the new test).

## Experiment: real matched OOD replay, E626's exact recipe, reused checkpoint

Reused E626's own already-trained scratch checkpoint verbatim — **no
retrain** — after verifying its sha256 matches E626's JSON exactly
(`c5b7c807…dd561221`, still present locally at
`outputs/runs/e626-required-slot-margin-scratch800-20260720/checkpoints/last.pt`).
Ran `scripts.evaluate_model` on the `ood` suite `n=4` with E626/E627's full
matched-recipe flags (`honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, `semantic_role_contract_in_context`,
`semantic_role_decode_weight=8`, `slot_coverage_close_decode_weight=2`,
`schema_value_decode_weight=4`, `schema_opaque_close_decode_weight=4`,
`schema_role_slot_decode_weight=8`, the full `semantic_plan_*` family at
E617's recorded weights, `grammar_ltr_max_tokens=160`), varying only
`required_slot_margin_decode_weight` over `{0, 2, 6}`.

| Metric | Control (0) | Margin=2 | Margin=6 |
| --- | ---: | ---: | ---: |
| meaningful v1 | 0.5000 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 | 0.0000 |
| placeholder fidelity | 0.5500 | 0.8333 | 0.8333 |
| placeholder validity | 0.7300 | 0.9000 | 0.9000 |
| structural similarity | 0.4886 | 0.5473 | 0.5473 |
| component recall | 0.4792 | 0.5625 | 0.5625 |
| reward | 0.8140 | 0.9005 | 0.9005 |
| AST node / edge F1 | 0.5437 / 0.3750 | 0.6137 / 0.3485 | 0.6137 / 0.3485 |
| `required_slot_margin` fires (frame_depth) | — | 17 (depths 1–3 only) | 17 (depths 1–3 only) |
| AgentV | 0/1 | 0/1 | 0/1 |

The `margin=0` control reproduces E626's own control exactly (all metrics
match to 4 decimal places, as does the fixture-only single-record AgentV
score), confirming a faithful replay before reading the treatment arms.

## Honest result

**The regression is fixed, not merely reduced, in this replay.** Margin=6
was E626's worst arm (every headline metric regressed well below control);
post-fix, margin=6 is **byte-identical** to margin=2's already-verified
positive arm — same per-record predictions across all 4 OOD records, same
`required_slot_margin_applications_sum` (17), same
`required_slot_margin_choice_changes_sum` (9), and the new
`constrained_selection_traces` confirm zero fires at `frame_depth == 0` in
either arm (all 17 fires land at `frame_depth` 1, 2, or 3 — inside an
already-open component/object frame, exactly the intended scope). Margin=2's
benefit fully survives the fix, matching E627's own prediction (E627 already
showed margin=2 never hit the `frame_depth == 0` failure mode, so it should
be unaffected — confirmed).

Both of E626's headline failure signatures are independently confirmed fixed
by inspecting the actual predicted programs, not just aggregate metrics:
Dashboard's predicted program at margin=6 is now
`root = Stack([v0, v1, v3, v4], "column")` with a real `Callout` and `Card`s
— not the bare `Button` E626 reported; Gallery's `ImageGallery` array is
`[{src: ":ood.gallery.img", alt: ":ood.gallery.alt"}]` — not the empty array
E626 reported re-closing at margin=6.

`binding_aware_meaningful_v2_rate_strict` (strict v2) stays `0.0` in every
arm, unchanged from E626 — this fix targets the structural root-hijack
mechanism, not the remaining `required_placeholder_missing` /
`placeholder_semantic_role_mismatch` failures.

This is a single small (`n=4`) matched-pair OOD replay on one reused scratch
checkpoint, not a confirmatory multi-seed/full-suite result. **Not a ship
claim.** No checkpoint was trained, promoted, or synced. It also does not
prove the fix is safe at *arbitrary* margin — only that it removes the
specific dose-dependent hijack E626/E627 observed within `{0, 2, 6}` on this
one checkpoint/suite; a wider sweep (e.g. 10, 20) or a different
checkpoint/seed could still surface a different failure mode once the floor
is large enough to dominate a legitimate frame_depth>=1 argument-position
competition, which was not tested here.

## Decision

Adopt the `frame_depth == 0` exclusion in `_required_slot_margin_bias` as a
real fix for the mechanism E626/E627 diagnosed, verified by identical
predictions and zero `frame_depth == 0` fires at both margin=2 and margin=6
in this replay. `required_slot_margin_decode_weight` remains default-off,
unchanged default, unchanged contract-gating. The lever's practical safety
envelope widens — margin dosage between 2 and 6 is no longer a live risk on
this evidence, though this is not yet a claim it is safe at any margin or on
a full suite.

## Next step (deferred)

1. E626's original next-step — a powered multi-seed replay sweeping
   `required_slot_margin_decode_weight` across a small grid on a full
   held-out/`rico_held` suite — remains open and untouched here; it is now
   reasonable to include margin values above 2 in that grid given this fix,
   though only demonstrated at `n=4` on one checkpoint so far.
2. This replay happened to make margin=2 and margin=6 fully identical; test
   whether a substantially larger margin (10, 20) or a different
   checkpoint/seed can still surface a *different* failure mode at
   `frame_depth >= 1` (competing against other legitimate slot-filling or
   array-closure biases inside an open frame) — not attempted here.
3. `binding_aware_meaningful_v2_rate_strict` remains 0.0 at every margin
   tested across E626/E627/E628; the coverage-aware component/property
   closure work E620 originally named is still the next lever after this
   one, not another margin/scope tuning pass on this same lever.

Raw evidence:
[JSON](iter-e628-required-slot-margin-frame-depth-exclusion-20260720.json).
