# E621 — the real `slot_contract_in_context` control arm, and a correction to E618/E619

Date: 2026-07-20
Status: completed, real scoring gap closed, no headline quality metric moved,
no new evaluator bug found, no code change, recipe-only fix; corrects an
overbroad claim carried from E618 into a concurrent E619 session

E618 found and fixed a real false positive in `binding_aware_meaningful_v2`'s
`binding_correctness` check, but `binding_aware_meaningful_v2_rate_strict`
("strict v2") stayed 0.0 on both arms of its matched replay, and E618 flagged
a **secondary, unfixed** gap as its own "next" step: none of the E611-E618 eval
recipes ever passed `--slot-contract-in-context` to `scripts.evaluate_model`,
so `required_inventory_coverage` — one of `binding_aware_meaningful_v2`'s 8
sub-checks — could never move past `CheckStatus.UNKNOWN`. This iteration
answers that directly, in the same instrument-first spirit as E617/E618.

**Note on numbering.** This work was drafted in parallel with a separate,
concurrent session that also answered E618's question and landed as
`## E619 chasing the --slot-contract-in-context gap` in
`docs/design/quality-experiment-matrix.md`
(`docs/design/iter-e619-slot-contract-in-context-gap-20260720.{md,json}`,
run IDs `e619-slotcontext-control-r1` / `e619-slotcontext-treatment-r1`).
Both sessions independently reach the same headline conclusion. This
iteration was renumbered once already, to E620, but a *second* concurrent
session also landed a commit on the same base branch under that number
(`## E620 the E617 bug class had a second, still-live instance`, a distinct
topic — a `semantic_plan_*` decode-weight gating gap, not
`slot_contract_in_context`) while this renumbering was in progress. Final
number is **E621**. It adds one real finding neither concurrent session had:
a genuinely live control arm for the `slot_contract_in_context` flag itself
(see "Correction to E618 and the concurrent E619" below).

## Method

Traced `slot_contract` end to end rather than assuming the write-up was
exactly right:

1. `GenerationRequest.from_record()` (`src/slm_training/data/contract.py`)
   always resolves `slot_contract` via `canonical_slot_contract(record.openui,
   declared=record.placeholders)` — i.e. the real, gold-program-derived
   placeholder inventory, regardless of any CLI flag.
2. `eval_runner._effective_request_for()`
   (`src/slm_training/harnesses/model_build/eval_runner.py:667`) unconditionally
   zeroes `data["slot_contract"]` unless `config.slot_contract_in_context` is
   `True`. This *effective*, possibly-zeroed request — not the raw one used
   for actual generation — is what gets passed to
   `binding_aware_meaningful_v2(pred, record=record,
   request=_effective_request_for(record))` for scoring.
3. Inside `meaningful_program.py`, `_prompt_contract()` sets
   `placeholder_coverage_known=bool(placeholders)`, where `placeholders`
   merges the prompt's own literal `Placeholders:`/`Inventory:` line
   (`record.prompt`, static, never mutated by generation-time prompt
   injection) with `request.slot_contract`. None of the four OOD prompts in
   `src/slm_training/resources/data/eval/remediated` carry a literal
   inventory line, so with the flag off, `placeholder_coverage_known` can
   only become `True` via `request.slot_contract` — which
   `_effective_request_for` zeroes.
4. `_inventory_check()`'s `required_inventory_coverage` verdict is `UNKNOWN`
   only when nothing else already forces a `FAIL` — its `role_mismatches`
   list is computed **unconditionally** from the predicted placeholders'
   identities (not gated on `contract.placeholder_coverage_known`), so a
   record can already be a real, judged `FAIL` even with the contract fully
   unknown.

Reused E616/E617's exact training recipe (this sandbox's `outputs/` started
empty again, same as every prior E615-E618 session) —
`--context-backend scratch --output-tokenizer choice --no-sync-checkpoints`,
same corpus (`src/slm_training/resources/data/train/e530_visible_semantic_roles_r2_20260719`,
244 records), same seed, 80 steps. The resulting checkpoint's sha256
(`119dd41a…8898a854`) and `last_loss` (26.5243) are byte-identical to
E616/E617's, confirming a true replay, not a different model.

Ran E617's exact corrected control recipe unchanged (`honest_slot_contract`,
`slot_contract_constrained_decode`, `schema_role_slot_decode_weight=0.0` — that
lever is not this iteration's variable) as control, and the identical recipe
plus `--slot-contract-in-context` as treatment, against the same checkpoint
SHA.

## Result

**The diagnosis was correct, and the fix is real.**
`binding_aware_meaningful_v2_coverage` rises **0.75 → 1.0**. Per-record
`required_inventory_coverage` status:

| record | control | treatment |
| --- | --- | --- |
| `ood_dashboard_01` | `FAIL` (`placeholder_semantic_role_mismatch`) | `FAIL` (`required_placeholder_missing`, `placeholder_semantic_role_mismatch`) |
| `ood_gallery_01` | **`UNKNOWN`** (`required_inventory_unknown`) | **`FAIL`** (`required_placeholder_missing`) |
| `ood_modal_01` | `FAIL` (`placeholder_semantic_role_mismatch`) | `FAIL` (`placeholder_semantic_role_mismatch`) |
| `ood_auth_01` | `FAIL` (`placeholder_semantic_role_mismatch`) | `FAIL` (`placeholder_semantic_role_mismatch`) |

Only `ood_gallery_01` moves — from an unjudged `UNKNOWN` to a real,
judged `FAIL`. `:ood.gallery.img` is genuinely missing from the prediction:
both `src` and `alt` collapse onto `:ood.gallery.alt`
(`schema_role_slot_decode_weight=0.0` in this recipe, the same collapse bug
E614/E615/E617 target on a different lever). This is a correct verdict given
the model's real output, not a false positive — **no further evaluator bug
was found this time**, unlike E618.

**No headline quality metric moves.** `parse_rate`, `syntax_parse_rate`,
`meaningful_program_rate`, `placeholder_fidelity`, `placeholder_validity`,
`structural_similarity`, `reward_score`, `ast_node_f1`, `ast_edge_f1`,
`component_type_recall`, and `exact_match` are bit-for-bit identical between
arms. `binding_aware_meaningful_v2_rate_strict` and
`_rate_coverage_conditioned` stay **0.0 → 0.0**: `ood_gallery_01`'s
newly-real verdict is `FAIL`, same as every other check on every other
record, so nothing newly *passes*. AgentV stays 0/1 in both arms.

**3 of 4 predictions are byte-identical.** `slot_contract_in_context` also
changes `_context_prompts`' context-encoder input (not just scoring
visibility — it feeds a second copy of the slot contract into
`_format_one_context`), so it can in principle perturb decode. On this
checkpoint it reaches only `ood_modal_01` — the one record E618 separately
flagged as a deeply malformed, self-nested prediction consistent with genuine
80-step-checkpoint undertraining — producing a single-character difference
deep inside an already-garbled string literal
(`...ettecmii=timeEm6QEte>me>=8:66em` vs `...ettecmii=timeEm6QEte>ee>=8:66em`).
It does not change `ood_modal_01`'s score; that record fails every check in
both arms regardless.

## Correction to E618 and the concurrent E619

E618's own secondary-finding text said `required_inventory_coverage` is
`UNKNOWN` **"for every OOD record"** across E611-E617. This iteration's real
per-record evidence shows that was overbroad: only `ood_gallery_01` was ever
`UNKNOWN`. The other three OOD records (`ood_dashboard_01`, `ood_modal_01`,
`ood_auth_01`) were already real, judged `FAIL`s the whole time via
`_inventory_check`'s `role_mismatches` path, which is computed unconditionally
from the predicted placeholders' identities and was never gated by the
`slot_contract_in_context` gap in the first place. The underlying diagnosis —
that the flag was needed for `ood_gallery_01`'s coverage check specifically —
still holds; the "every record" framing does not.

The concurrent E619 session repeats this same overbroad claim ("moves from
`UNKNOWN` to a real judged verdict for all 4 records"), and its own committed
JSON shows why: its `eval_recipe.slot_contract_in_context` is a single flat
`true` applied to *both* its `control` and `treatment` runs (its
`control.binding_aware_meaningful_v2_coverage` is already `1.0`, not `0.75`).
The only lever E619 varied live was `schema_role_slot_decode_weight` (an
unrelated E614/E615/E617 lever, 0 vs 8); its "before" state
(`UNKNOWN` for all 4) was carried over from E618's prose, not measured live
with `--slot-contract-in-context` genuinely unset in that session. This
iteration's control run (flag genuinely unset, same checkpoint class) is the
first live measurement of the true pre-flag baseline, and it disagrees with
both write-ups on scope while agreeing with both on the flag's real,
measured effect (the post-flag/treatment numbers match: coverage `1.0`,
identical headline metrics, the same one-character `ood_modal_01` noise
diff).

## Honesty check: does this reintroduce a gold-placeholder channel?

`GenerationRequest.slot_contract` is derived from
`canonical_slot_contract(record.openui, declared=record.placeholders)` — the
**gold** program — which on its face looks exactly like the pattern
`CLAUDE.md`'s contract warns against ("never reintroduce silent
`gold.placeholders` channels under `honest_slot_contract=True`"). It is not a
new or silent channel here, though: `honest_slot_contract=True` (already set
throughout E611-E618) already makes `TwoTowerModel.generate_batch_requests()`
surface this exact same list into the literal, model-visible prompt text via
`ensure_prompt_inventory()` **before generation**, independent of
`slot_contract_in_context`. `slot_contract_in_context` only controls (a)
whether that already-visible list is *also* embedded a second time into the
context-encoder's input text (`_context_prompts`), and (b) whether the scorer
is allowed to treat it as "required" for `required_inventory_coverage`.
Enabling it does not give the model any information it did not already have
under `honest_slot_contract=True`; it only makes the scorer's judgment
consistent with the model's real input.

## Decision

Recommend `--slot-contract-in-context` join the standard eval recipe
alongside E617's `--slot-contract-constrained-decode` for any future replay
of this lineage — it closes a real (if narrow) unjudged-verdict gap with no
honesty regression and no observed quality-metric change on this checkpoint
class. **Not a promotion or ship claim**: this is still an `n=4` diagnostic
scratch-checkpoint replay, `binding_aware_meaningful_v2_rate_strict` remains
0.0 in both arms, and AgentV is 0/1 in both arms.

No production source code changed this iteration — `python -m
scripts.verify_version_stamps --check` confirms 0 components touched
(matching E616's pattern: this is an eval-recipe finding, not a code fix).
One new regression test was added,
`tests/test_harnesses/model_build/test_eval_gates.py::test_slot_contract_in_context_turns_required_inventory_coverage_from_unknown_into_a_real_verdict`,
using the existing `evaluate()` + stub-model harness pattern to protect the
exact `UNKNOWN → real-verdict` transition observed here (a minimal record
whose prompt mentions its one required component by name but not its
placeholder inventory, so toggling the flag is the only thing that can move
`required_inventory_coverage` off `UNKNOWN`).

## Next

`binding_aware_meaningful_v2_rate_strict` is still 0.0 in both arms on this
80-step scratch checkpoint for reasons unrelated to `slot_contract_in_context`:
`ood_dashboard_01`/`ood_modal_01` fail `schema_value_role_correctness` and
`anti_gaming`, `ood_gallery_01` now fails `required_inventory_coverage` for
real (the known `schema_role_slot_decode_weight=0` collapse bug), and
`ood_auth_01`/`ood_dashboard_01` fail `required_inventory_coverage` on role
mismatches. None of these look like evaluator artifacts on this evidence. The
next well-motivated replay is E614/E615/E617's own recipe
(`schema_role_slot_decode_weight` control-vs-treatment) with
`--slot-contract-in-context` *also* added, ideally on a real
trained/warm-started (non-scratch) checkpoint, to see whether closing this
scoring gap changes anything once the Gallery `src`/`alt` collapse itself is
also fixed by the treatment arm.

Evidence: [JSON](iter-e621-slot-contract-in-context-control-20260720.json).
Concurrent sessions on the same base branch: [E619 narrative](iter-e619-slot-contract-in-context-gap-20260720.md)
/ [JSON](iter-e619-slot-contract-in-context-gap-20260720.json) (same question,
no live control) and [E620 narrative](iter-e620-semantic-plan-weight-gate-gap-20260720.md)
/ [JSON](iter-e620-semantic-plan-weight-gate-gap-20260720.json) (a different
topic: a `semantic_plan_*` decode-weight gating gap).
