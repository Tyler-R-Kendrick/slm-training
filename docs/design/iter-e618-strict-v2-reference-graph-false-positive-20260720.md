# E618 — a real false positive in the strict meaning-v2 evaluator (and why strict-v2 still stays 0)

Date: 2026-07-20
Status: completed, real evaluator bug found and fixed, strict v2 metric still
0.0 on E617's checkpoint for separate, genuine reasons

E617 found a real, positive treatment-vs-control quality delta on a matched
OOD `n=4` replay (`placeholder_fidelity` +0.042, `placeholder_validity`
+0.025, `reward_score` +0.0125) but `binding_aware_meaningful_v2_rate_strict`
("strict v2") stayed 0.0 in both arms and asked, as its own "next" question,
whether that 0 reflects a real quality gap or another silent misconfiguration.
This iteration answers that directly for the `_binding_check` half of strict
v2, in the same spirit as E617's slot-contract-decode finding: instrument the
real evaluator against the real checkpoint's real predictions and see what
actually fails.

## Method

Rather than retrain, this session reused the real, on-disk E617 eval
artifacts (`outputs/runs/e617-control-eval-r1/eval_ood.json` and
`.../e617-treatment-eval-r1/eval_ood.json`, which persisted in this sandbox)
— same checkpoint sha256 (`119dd41a…8898a854`), same 4 real predictions per
arm. Called `binding_aware_meaningful_v2()`
(`src/slm_training/evals/meaningful_program.py`) directly against each real
prediction and printed every one of its 8 sub-checks' status and reason
codes.

**Result:** `ood_gallery_01` and `ood_modal_01` both failed the
`binding_correctness` check with reason `reference_graph_invalid`, in both
arms.

## Root cause

`_binding_check` already used the official parser's own `unresolved`/
`orphaned` analysis (authoritative for both structural placeholder scaffolds
and runtime `$state`/`Query`/`Mutation` programs) — but for any source
*without* `$`/`Query`/`Mutation`/`Action`/`@` syntax, it *also* ran a second,
redundant regex-based fallback: `evaluate_gate(Gate.REFERENCES, …)`
(`src/slm_training/data/verify/stack.py`'s `_reference_graph`). That
regex-based gate strips quoted strings but not bare identifiers, so it
misreads bare object-literal property keys — `src:`/`alt:` in a typed-array
item like `{src: ":ood.gallery.img", alt: ":ood.gallery.alt"}`, exactly the
syntax E614/E615/E617's own object-frame lineage targets — as unresolved
variable references.

Standalone repro:

```
evaluate_gate(Gate.REFERENCES, record_with(
    'root = Stack([v0], "column")\nv0 = ImageGallery([{src: ":ood.gallery.img", alt: ":ood.gallery.alt"}])'
))
# -> GateStatus.FAIL, detail "unresolved reference: alt"
```

even though the real official parser (`slm_training.dsl.parser.parse`/
`validate`, the actual grammar) accepts this source cleanly and reports zero
unresolved/orphaned bindings in `program.meta`.

Grepping `docs/design/iter-e612..e617*.json` shows `reference_graph_invalid`
already present in **E612, E613, and E614's own eval evidence** — this is not
new to E617; it has been silently pinning strict v2 toward 0 across at least
six prior experiments in this lineage, predating the E617 decode-gap fix
entirely.

## Fix

`_binding_check` no longer branches on runtime-syntax presence. It always
runs the graph-based dependency-reachability pass previously reserved for
runtime sources (walk `$state`/`Query`/`Mutation` dependencies, report
`unreachable_binding` for genuinely dead runtime bindings). For
structural-only sources, `state_declarations`/`query_statements`/
`mutation_statements` are empty, so this is a verified no-op — structural
correctness is already fully covered by the official-parser-derived
`unresolved`/`orphaned` checks that already ran unconditionally above it. The
buggy regex fallback and its `reference_graph_invalid` reason are removed
from this call site entirely; `src/slm_training/data/verify/stack.py`'s
shared `Gate.REFERENCES`/`_reference_graph` implementation is untouched
(still used elsewhere in the verify stack for its original purpose).

`evals.meaningful_program` version bumped `2.0.0` → `2.1.0`
(`src/slm_training/resources/versions.json`). New regression test:
`tests/test_evals/test_meaningful_program.py::test_v2_typed_array_object_item_keys_are_not_treated_as_unresolved_refs`.

## Honest result

Re-scored all 4 real E617 OOD predictions, before and after the fix
(before-state obtained via `git stash push` on only `meaningful_program.py`,
re-running against the same real predictions, then `git stash pop`):

- **`reference_graph_invalid` disappears** from `ood_gallery_01` and
  `ood_modal_01` in both arms — a real, confirmed false positive removed.
- **`binding_aware_meaningful_v2_rate_strict` stays 0.0 → 0.0** in both arms.
  `ood_gallery_01` still fails on `required_inventory_unknown` (a separate,
  unfixed gap — see below). `ood_modal_01` still fails on a long list of
  `schema_value_role_mismatch:*` / `placeholder_semantic_role_mismatch` /
  `duplicate_subtree_spam` / `placeholder_spam` reasons — its raw prediction
  is a deeply malformed, self-nested `Modal`/`Stack` tree, consistent with an
  undertrained 80-step CPU scratch checkpoint, not an evaluator artifact.
  `ood_dashboard_01` and `ood_auth_01` never triggered the bug at all (no
  object-literal syntax in their predictions) and fail on real schema/
  placeholder-role mismatches, unaffected either way.

**This is a genuine, checkpoint-quality-driven 0 for this specific 80-step
diagnostic run, once the evaluator bug is no longer in the way.** The fix is
real and worth keeping — it was actively suppressing a true signal in
E612-E614 as well — but it does not, on its own, unlock strict v2 for this
checkpoint class.

## Secondary finding (not fixed this iteration)

`required_inventory_coverage` is `CheckStatus.UNKNOWN` (never a real judged
PASS/FAIL) for all 4 OOD records, in both pre- and post-fix runs, because none
of the E611-E617 recipes ever set `--slot-contract-in-context`.
`_effective_request_for()` (`harnesses/model_build/eval_runner.py`) zeroes
`GenerationRequest.slot_contract` unless `config.slot_contract_in_context` is
true, and that same effective request feeds `binding_aware_meaningful_v2`'s
`_prompt_contract()`. None of these OOD prompts contain a literal
`placeholders:` inventory line either, so `placeholder_coverage_known` is
`False` and the check degrades to `UNKNOWN`. `UNKNOWN` fails the verdict
exactly like `FAIL`, so this alone doesn't explain today's 0 (the other
checks already fail on their own), but it means `required_inventory_coverage`
has never actually been exercised as a real judged check in this recipe —
structurally the same shape as E617's finding (an orthogonal flag nobody
sets, silently degrading a metric), just landing on `UNKNOWN` instead of a
hard no-op. Left open, flagged as the clear next step, to keep this
iteration's scope to the one `binding_correctness` bug.

## Decision

Keep the `_binding_check` fix — it removes a confirmed, real false positive
present in eval evidence since at least E612. Do not claim it moves strict v2
off 0: for this exact checkpoint's real predictions, it honestly does not.
The remaining failures are genuine 80-step-checkpoint quality gaps
(malformed `ood_modal_01` output, schema/placeholder role mismatches) plus
the still-open `required_inventory_coverage`/`slot_contract_in_context` gap.
No checkpoint trained, promoted, or synced this iteration; not a ship claim.

## Next

Try `--slot-contract-in-context` on a matched pair to see whether
`required_inventory_coverage` starts reporting real judgments instead of
`UNKNOWN`, and whether that — combined with this fix — moves strict v2 at all
on a real (ideally non-scratch) checkpoint. Separately, `ood_modal_01`'s raw
prediction is worth a closer look on its own merits as a checkpoint-capacity/
decode-stability question.

Evidence: [JSON](iter-e618-strict-v2-reference-graph-false-positive-20260720.json).
