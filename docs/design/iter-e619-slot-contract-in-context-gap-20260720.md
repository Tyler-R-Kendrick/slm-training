# E619 — chasing the `--slot-contract-in-context` gap (safe, real, doesn't flip strict v2)

Date: 2026-07-20
Status: completed, real matched control/treatment eval, real (negative)
answer to E618's open question

E618 fixed a real evaluator false positive but left one question open as its
"next" note: `required_inventory_coverage` degrades to `CheckStatus.UNKNOWN`
(never a real judged PASS/FAIL) for every OOD record because none of the
E611-E618 eval recipes ever set `--slot-contract-in-context`. This iteration
chases that gap directly, in the same spirit as E617's slot-contract-decode
finding: is this an orthogonal flag nobody sets, silently degrading a metric —
and if so, does fixing it move `binding_aware_meaningful_v2_rate_strict` off
0.0 on any of the 4 real OOD records?

## Step 1: is this safe to add?

Before touching the eval recipe, this session checked whether
`--slot-contract-in-context` changes the task unfairly (e.g. leaking the gold
placeholder inventory into the model's own generation context) rather than
being a legitimate eval-time override.

- `apply_runtime_overrides` (`src/slm_training/harnesses/model_build/factory.py`,
  explicitly documented as "Apply eval/train decode + conditioning overrides
  onto a loaded plugin") lists `slot_contract_in_context` among its
  overridable fields — the same category as `slot_contract_constrained_decode`,
  the flag E617 already confirmed is a legitimate, intended eval-time-only
  override that requires no retraining.
- `_resolve_slot_contract` (`src/slm_training/models/twotower.py`) — the
  function that actually builds the `---SLOT_CONTRACT---` context block fed to
  the model when `slot_contract_in_context=True` — only falls back to
  `gold.placeholders` when `honest_slot_contract=False`. Every E611-E618
  recipe already sets `honest_slot_contract=True`, so the code path actually
  exercised derives the inventory *only* from `inventory_from_prompt(prompt,
  design_md, heuristic=True)` — the user-visible prompt/DESIGN.md text, never
  the gold answer.

**Conclusion: safe.** Turning this on does not leak gold labels; it surfaces
an inventory honestly derivable from the prompt itself, which is the intended
production behavior of the honest-slot-contract system. This was also
verified empirically (see below): predictions came back byte-identical to the
E617 baseline on 3 of 4 records.

One honest caveat: the checkpoint itself was trained *without*
`slot_contract_in_context=True` (no E611-E618 training recipe set it), so this
is a genuine train/eval configuration difference, not a like-for-like replay
of training conditions. The byte-identical-prediction result below suggests
it didn't matter for this checkpoint's decode path on these records, but that
should not be assumed to generalize to other checkpoints.

## Method

No retraining. Reused the real, on-disk E617 checkpoint
(`outputs/runs/e617-debug-repro-scratch80-20260720/checkpoints/last.pt`,
sha256 `119dd41a…8898a854`, verified via `sha256sum` before running) and
replayed E617's exact matched control (`schema_role_slot_decode_weight=0.0`)
vs treatment (`=8.0`) recipe through `python -m scripts.evaluate_model`,
reading every flag/weight directly from E617's own recorded
`evaluation_policy` block and adding only `--slot-contract-in-context`. Unlike
E618 (which re-scored existing predictions), this required **real fresh
generation** — the flag changes what's fed to the model — so two real eval
runs (control + treatment) were executed against the real checkpoint, each
completing within the sandbox's per-command budget.

## Result

**Headline metrics: unchanged.** `parse_rate`, `syntax_parse_rate`,
`meaningful_program_rate`, `placeholder_fidelity`, `placeholder_validity`,
`reward_score`, `structural_similarity`, `ast_node_f1` are identical to E617's
values in both arms. 3 of 4 OOD predictions (`ood_dashboard_01`,
`ood_gallery_01`, `ood_auth_01`) are byte-identical to E617's. `ood_modal_01`
differs by a single character deep inside its already-malformed, self-nested
garbage tail (`...QEte>ee>=8:66em` vs `...QEte>me>=8:66em`), with no metric or
check-status impact — consistent with that record's output being decode-level
instability on a pathological tree, not something this flag materially
changed.

**`required_inventory_coverage` moves from `UNKNOWN` to a real, judged
verdict for all 4 records, in both arms.** `binding_aware_meaningful_v2_coverage`
(fraction of records with a known, non-`UNKNOWN` coverage verdict) rises
0.75 → 1.0 in both control and treatment. This is exactly E618's hypothesis,
confirmed with a real eval:

| Record | Verdict (both arms) | Reason codes |
| --- | --- | --- |
| `ood_gallery_01` | FAIL | `required_placeholder_missing` — missing `:ood.gallery.caption`, `:ood.gallery.cta`, `:ood.gallery.hint.body`, `:ood.gallery.hint.title`, `:ood.gallery.img` |
| `ood_dashboard_01` | FAIL | `required_placeholder_missing`, `placeholder_semantic_role_mismatch` |
| `ood_modal_01` | FAIL | `placeholder_semantic_role_mismatch` |
| `ood_auth_01` | FAIL | `placeholder_semantic_role_mismatch` |

`binding_correctness` still PASSes for all 4 records in both arms — E618's
fix holds under this recipe too.

**`binding_aware_meaningful_v2_rate_strict` stays 0.0 → 0.0 in both arms — no
item flips to a strict-v2 pass.** All 4 records now fail
`required_inventory_coverage` for genuine content reasons (the 80-step
checkpoint's predictions really are missing required placeholders or using
mismatched semantic roles), not because the check silently no-oped. This is a
real, honest 0 — E618's open coverage gap is now fully accounted for, not a
promotion and not a ship-gate-clearing result.

## Decision

Add `--slot-contract-in-context` to future E611-E619-lineage recipes going
forward: it's a strict improvement in judgment honesty (no more silently
degrading `required_inventory_coverage` to `UNKNOWN`) at no measured cost to
headline metrics or predictions on this checkpoint. Do not claim this moves
strict v2 off 0 — for this exact checkpoint's real predictions, it honestly
does not; the checkpoint itself does not yet emit the full required
placeholder set. No code changed this iteration (`python -m
scripts.verify_version_stamps --check` confirms 0 components touched). No
checkpoint trained, promoted, or synced. Not a ship claim.

## Next

Replay this same matched pair on a more-trained (non-80-step-scratch)
checkpoint to see whether `required_inventory_coverage` starts PASSing once
the model has learned to emit the full placeholder set, and whether
`binding_aware_meaningful_v2_rate_strict` finally moves off 0.0 once
checkpoint capability — not eval configuration — is the only remaining
variable. Separately, `ood_modal_01`'s raw prediction remains worth a closer,
checkpoint-capacity-focused look (E618's still-open second candidate); this
iteration's single-character noise diff between two otherwise-identical runs
is a small extra data point that its self-nested garbage output is
decode-level instability, not a fixed/deterministic bug.

Evidence: [JSON](iter-e619-slot-contract-in-context-gap-20260720.json).
