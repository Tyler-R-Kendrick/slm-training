# E617 — the slot-contract decode gap upstream of E612/E615/E616

Date: 2026-07-20
Status: completed, positive real end-to-end result, code retained, checkpoint
rejected, not promotable

E616 replayed E615's matched OOD `n=4` control (`schema_role_slot_decode_weight=0`)
vs treatment (`=8.0`) eval on a fresh 80-step scratch checkpoint and found both
arms byte-identical: `ood_gallery_01` decodes to `ImageGallery([])` in both
arms, because the typed array closes empty before an item object is ever
opened — E615's object-frame lever's precondition is never reached. E616
attributed this to reproducing E612's already-rejected "undertrained
checkpoints close typed arrays empty" finding, one level upstream of E615, and
proposed as its own "next" step: "revisit why
`semantic_plan_typed_array_nonempty_margin_decode_weight=2.0` does not prevent
an empty-array close on this checkpoint class." This iteration answers that
question directly.

## Method

Rather than assume the answer was "needs more training" (E612/E616's working
theory), this session added temporary instrumentation inside
`_semantic_plan_typed_array_nonempty_bias` (`src/slm_training/models/twotower.py`,
reverted before commit — not part of the shipped diff) to print which internal
gate returns `None` on every call. Retrained the identical E616 recipe
(same corpus/seed/steps/architecture, 80 scratch steps) — the resulting
checkpoint's sha256 is byte-identical to E616's, confirming a true replay, not
a different model. Reran E616's exact control eval with the instrumentation
active.

**Result: 111/111 calls into the array-decision branch failed the very first
gate — `not slot_contract` — because `self._slot_contracts[row]` was `None`.**
Not the owner-lookup gate, not the schema-reachability gate, not the
visible-slot-exhaustion gate: the *very first* precondition the E612 lever
checks was already unmet, for every OOD record, at every decode step.

## Root cause

`self._slot_contracts` (in `TwoTowerModel._generate_batch_once`) is only
populated when `config.slot_contract_constrained_decode` or
`config.template_fill_decode` is `true`. Checking all four surviving
`eval_recipe` JSON blocks in `docs/design/iter-e611*.json` through
`iter-e616*.json` confirms neither flag was ever passed. The recipe set
`honest_slot_contract=true` and nonzero weights for five slot-contract-gated
decode biases:

- `schema_role_slot_decode_weight` (`_schema_role_slot_bias`, E591/E615)
- `slot_coverage_close_decode_weight` (`_slot_coverage_close_bias`, E614)
- `semantic_plan_typed_array_nonempty_margin_decode_weight` (E612)
- `semantic_plan_typed_array_item_margin_decode_weight` (E613)
- `semantic_plan_repeated_slot_margin_decode_weight` (E611)

All five gate on the identical `not slot_contract` check. `honest_slot_contract`
and `slot_contract_constrained_decode` are orthogonal flags — one governs
*where the inventory comes from* (honest prompt-only vs. legacy gold
fallback), the other governs *whether the resolved inventory is plumbed into
decode-time biases at all* — and the E611-E616 recipe conflated them. Every
one of these five biases has been silently returning `None` on every decode
step across six experiments, independent of weight value or checkpoint
quality. This reframes E612's and E616's "empty array close" finding: it
wasn't (only) that the checkpoint was too weak to prefer opening an item —
the very mechanism meant to make that preference possible was never wired in.

## Fix

Two parts, both small and scoped:

1. **Recipe correction.** Replayed the identical E616 matched control/treatment
   pair with one flag added — `--slot-contract-constrained-decode` — and no
   other value changed, against the same checkpoint sha256.
2. **Fail-loud guard** (`src/slm_training/models/twotower.py`,
   `_generate_batch_once`, `model.twotower` v58→v59): raise `ValueError`
   naming the offending weight(s) when any of the five contract-gated decode
   weights is set without `slot_contract_constrained_decode` or
   `template_fill_decode`. This turns the E611-E616 no-op into a hard failure
   for any future recipe, instead of a silent, expensive-to-diagnose gap.
   Verified with 7 new parametrized unit tests in
   `tests/test_models/test_compiler_decode.py`
   (`test_contract_gated_decode_weight_without_slot_contract_decode_raises`,
   `test_contract_gated_decode_weight_with_slot_contract_decode_does_not_raise`).

## Real end-to-end result

With the flag corrected, **all 4 of 4 OOD predictions differ between control
and treatment** — the first time in the E611-E616 lineage a same-checkpoint
matched pair has not been byte-identical. `ood_gallery_01`:

- control (`schema_role_slot_decode_weight=0`):
  `ImageGallery([{src: ":ood.gallery.alt", alt: ":ood.gallery.alt"}])` — the
  array now opens an item (E612's fix finally executes), but `src` and `alt`
  collapse onto the same slot (`:ood.gallery.alt`) — this is exactly E614's
  narrative bug.
- treatment (`schema_role_slot_decode_weight=8.0`):
  `ImageGallery([{src: ":ood.gallery.img", alt: ":ood.gallery.alt"}])` — `src`
  and `alt` each bind their own matching visible slot. This is E615's fix,
  demonstrated end-to-end for the first time on a real checkpoint and a real
  eval, not just at the unit level.

Aggregate deltas (treatment − control, same checkpoint sha256
`119dd41a…8898a854`): `placeholder_fidelity` 0.7417→0.7833 (+0.042),
`placeholder_validity` 0.845→0.87 (+0.025), `reward_score` 0.6493→0.6618
(+0.0125), `structural_similarity`/`tree_edit_similarity` 0.5509→0.5548
(+0.0038), `ast_node_f1` 0.6002→0.6047 (+0.0044). `parse_rate`,
`syntax_parse_rate` (both 1.0), `meaningful_program_rate` (0.5),
`binding_aware_meaningful_v2_rate_strict` (0.0), `component_type_recall`
(0.6875), and `ast_edge_f1` (0.4167) are unchanged. Strict v2 remains 0 in
both arms; this is not a promotion or ship result, and this single small
(`n=4`) replay is diagnostic, not confirmatory. No checkpoint was created for
promotion, and none was synced.

## Decision

Retain the E617 `ValueError` guard as permanent, default-on safety wiring (no
config flag; it only fires when contract-gated weights are already set
without their required companion). Retain E611-E616's code unchanged — the
bug was in the eval recipe, not in any of those levers' own logic. Reject this
session's checkpoint for promotion (80-step CPU scratch, strict v2 still 0).
The next matched-pair session for any of E611/E612/E613/E615 should include
`--slot-contract-constrained-decode` and, ideally, run against a real
trained/warm-started checkpoint rather than another 80-step scratch model, to
see whether these real (if small) deltas hold or grow at scale.

Evidence: [JSON](iter-e617-slot-contract-decode-gap-20260720.json).
