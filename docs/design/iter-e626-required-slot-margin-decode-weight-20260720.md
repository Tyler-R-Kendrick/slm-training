# E626 — a decode-time margin that floors still-missing required slots directly

Date: 2026-07-20
Status: real matched OOD replay; honest dose-dependent result; not a ship claim

## Hypothesis

E619 -> E620 converged on: duration-only scaling of `schema_role_slot_decode_weight`
does not fix required-slot coverage; "next work should target coverage-aware
component/property closure". An open (unmerged) PR #625 stated the next lever
more concretely: floor still-missing *required slots* directly, the same way
`semantic_plan_margin_decode_weight` already floors still-required *plan
families* above the best legal component score. E626 implements that lever
independently and measures it.

## Implementation

A new field, `required_slot_margin_decode_weight` (default `0.0`, off), was
added end to end following the repo's existing `*_decode_weight` convention,
studied from `schema_role_slot_decode_weight`, `slot_coverage_close_decode_weight`,
and `semantic_plan_margin_decode_weight`:

- `TwoTowerConfig.required_slot_margin_decode_weight` (`src/slm_training/models/twotower.py`)
- `TwoTowerModel._required_slot_margin_bias`: among the current position's
  grammar-legal candidates, finds visible-slot tokens whose slot has not
  appeared anywhere in the emitted prefix yet, and floors the best-scoring
  still-missing one to `max(legal candidate scores) + margin` — the same
  flooring pattern `_semantic_plan_margin_decode_weight` uses for missing plan
  families, and the same "unused visible slot" bookkeeping
  `_semantic_plan_repeated_slot_bias` uses, but scoped to the *whole* prefix's
  slot-contract coverage rather than one repeated component span. Unlike
  `_schema_role_slot_bias` (a flat bonus for any role-compatible slot, filled
  or not), this only fires for slots `required_inventory_coverage` would
  otherwise still judge missing, and is a no-op once every slot in the legal
  candidate set has already been emitted.
- Wired into the decode loop right after `_slot_coverage_close_bias`.
- Added to the E617 `_generate_batch_once` `ValueError` guard list, since it
  reads `self._slot_contracts[row]` and would otherwise silently no-op without
  `--slot-contract-constrained-decode` / `--template-fill-decode` (the exact
  footgun E617 closed for the other five contract-gated weights).
- `ModelBuildConfig.required_slot_margin_decode_weight`, `apply_runtime_overrides`'s
  allowed-field list, and `_effective_evaluation_policy` were extended to
  match.
- CLI: `--required-slot-margin-decode-weight` added to both
  `scripts/evaluate_model.py` (matching the existing decode-weight flag style)
  and `scripts/train_model.py`. At this branch tip, `train_model.py`'s CLI had
  *no* `--semantic-plan-*` / `--schema-*` decode-weight flags at all (PR #623's
  claimed backfill has not landed here), so only this one new flag was added
  to `train_model.py`, matching `evaluate_model.py`'s established style rather
  than attempting to backfill the whole missing family.
- `model.twotower` v59 -> v60, `harness.model_build.eval` v32 -> v33
  (`python -m scripts.verify_version_stamps --check`: ok).

## Unit test

`tests/test_models/test_compiler_decode.py::test_required_slot_margin_bias_floors_only_still_missing_slots`
proves: (1) the bias floors the best-scoring legal slot candidate that is
genuinely still missing from the prefix; (2) it is a no-op once every visible
slot in the legal candidate set has already appeared; (3) the default-off
weight (`0.0`) never fires even with missing slots present; (4) it is a no-op
with no slot contract. It was also added to the existing parametrized
`test_contract_gated_decode_weight_without_slot_contract_decode_raises` guard.
121 tests pass in `test_compiler_decode.py` + `test_choice_tokenizer.py`; the
209 tests in `tests/test_harnesses/model_build` pass (one pre-existing,
unrelated `test_vocab_is_fixed_and_typed` failure — confirmed present on `main`
before this change too, via `git stash`).

## Experiment: real train + real matched OOD replay

Reused E620's exact scratch recipe verbatim (E530-r2 corpus, 244 records,
`context_backend=scratch`, `output_tokenizer=choice`, seed 0, batch size 1, 800
steps, `--no-sync-checkpoints`). The fresh checkpoint's `last_loss`
(4.068013…) matches E620's (4.068010…) to 4 decimal places, confirming an
equivalent scratch checkpoint. Wall time: 30.2 s (well under the 3-minute cap).
Checkpoint sha256 `c5b7c807…dd561221` (local-only, not synced, not promoted).

Eval recipe replayed the full E617-era accumulated lineage recipe (not just
E620's two flags) — `honest_slot_contract`, `slot_contract_constrained_decode`,
`slot_contract_in_context`, `semantic_role_contract_in_context`,
`semantic_role_decode_weight=8`, `slot_coverage_close_decode_weight=2`,
`schema_value_decode_weight=4`, `schema_opaque_close_decode_weight=4`,
`schema_role_slot_decode_weight=8` (fixed in every arm), and the full
`semantic_plan_*` family at the weights recorded in E617's own JSON — plus
`grammar_ltr_max_tokens=160`, on `src/slm_training/resources/data/eval/remediated`
`ood` suite `n=4`. The `required_slot_margin_decode_weight=0` control run
reproduces E620's treatment numbers and per-record predictions exactly
(fidelity 0.5500, validity 0.7300, structure 0.4886, reward 0.8140), confirming
the replay is faithful before adding the new lever.

Two treatment arms varied only `required_slot_margin_decode_weight`:

| Metric | Control (0) | Treatment margin=2 | Treatment margin=6 |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 1.0000 |
| meaningful v1 | 0.5000 | 0.7500 | 0.2500 |
| strict meaning v2 | 0.0000 | 0.0000 | 0.0000 |
| v2 coverage | 1.0000 | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5500 | 0.8333 | 0.3000 |
| placeholder validity | 0.7300 | 0.9000 | 0.4800 |
| structural similarity | 0.4886 | 0.5473 | 0.4261 |
| component recall | 0.4792 | 0.5625 | 0.3958 |
| reward | 0.8140 | 0.9005 | 0.5693 |
| AST node / edge F1 | 0.5437 / 0.3750 | 0.6137 / 0.3485 | 0.4770 / 0.2500 |
| latency p50 / p95 (ms) | 1289.5 / 7235.8 | 1117.3 / 4755.8 | 1120.8 / 10454.4 |
| AgentV | 0/1 | 0/1 | 0/1 |

No decode timeouts or fallbacks in any arm.

## Honest result

**Dose-dependent, real effect.** At a moderate margin (2.0 — the same scale as
the other `*_margin_decode_weight` levers already active in this recipe:
`semantic_plan_margin_decode_weight`, `semantic_plan_root_margin_decode_weight`,
etc.), the new lever raises every headline quality metric except AST edge F1:
Dashboard gains a `Callout` + `Card`s it previously omitted, Auth gains a
correctly role-matched `Button`/`Input` pair instead of a single
overloaded `Button`. At a strong margin (6.0), the same lever instead
*regresses* every headline metric well below control — Dashboard collapses to
a single bare `Button`, and Gallery's typed array closes empty again (E612's
already-rejected failure mode reappearing). This shows an overly large floor
can hijack early root/component-choice decisions away from correct component
selection, not just fill slots — a genuinely new, useful negative data point
in its own right.

`binding_aware_meaningful_v2_rate_strict` (strict v2) stays `0.0` in all three
arms: at least one record in every arm still fails
`required_placeholder_missing` and/or `placeholder_semantic_role_mismatch`.
This checkpoint does not clear strict meaning even at the
best-performing margin — real progress on the intermediate metrics, not a
strict-v2 pass.

This is a single small (`n=4`) matched-pair OOD replay on one scratch
checkpoint, not a confirmatory multi-seed/full-suite result. **Not a ship
claim.** No checkpoint was promoted or synced.

## Decision

Retain `required_slot_margin_decode_weight` as a promising default-off lever
at a moderate margin scale (~2.0, matching sibling margin weights already in
the recipe). Do not adopt margin=6 as a default — it is demonstrably worse
than turning the lever off. Keep the lever off by default in any shipped
recipe until a powered replay confirms it on a full suite.

## Next step (deferred)

1. Run a powered/confirmatory multi-seed replay (H19's protocol) sweeping
   `required_slot_margin_decode_weight` across a small grid (e.g. 1, 2, 3, 4)
   on a full held-out/`rico_held` suite, not just `n=4` OOD, before considering
   this margin for a default recipe change.
2. Root-cause *why* margin=6 hijacks the Dashboard root-component choice
   (Button instead of Card/Stack) and re-empties Gallery's typed array — is
   the bias only supposed to compete with other slot candidates, or is it
   interacting with another lever's flooring at the same decode position? A
   decode-time bias regressing structure by over-flooring a slot candidate
   above a correct component candidate is itself worth a root-cause iteration,
   in the spirit of E616/E617's upstream-gap chases.
3. `binding_aware_meaningful_v2_rate_strict` remains 0.0 at every margin tested
   here; the next lever after tuning this one should keep targeting the
   remaining `required_placeholder_missing` / `placeholder_semantic_role_mismatch`
   failures directly (coverage-aware component/property closure, as E620 named
   it), not another duration/margin replay of the same lever.

Raw evidence:
[JSON](iter-e626-required-slot-margin-decode-weight-20260720.json).
