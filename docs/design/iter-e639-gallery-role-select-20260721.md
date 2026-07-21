# E639 — Gallery positive role-select

Date: 2026-07-21
Status: completed inconclusive; checkpoint-divergence blocked; default-off; not ship

E638 (open, unmerged PR #649) tried a blanket "require complete visible
slot-contract coverage before semantic-plan root closure" gate. It collapsed
Gallery from a 3-slot `ImageGallery(img, alt, caption)` down to a single
`TextContent(":ood.gallery.hint.title")` — refusing to close root gave the
decoder no structurally compatible next family, so it abandoned the whole
component instead. E638's own conclusion: a future Gallery lever must
*positively select* the missing role-compatible sibling component instead of
only preventing closure, and it must be scoped so it cannot touch Modal or
Auth. E639 builds that lever.

## What was built

`src/slm_training/models/twotower.py::_gallery_role_select_bias`, gated by a
new default-off `gallery_role_select_decode_weight` (0.0 unless explicitly
set). Unlike E638's gate, it never biases the array's own close token — it
only raises the score of a *specific* missing role-compatible sibling
component (e.g. the hint `TextContent`) when one is legally available, so the
Stack array can still close normally if no compatible sibling exists. It is
structurally incapable of firing for any family other than ImageGallery: it
requires (a) the active frame to be the array directly owned by the root
Stack, and (b) an `ImageGallery` component to already have been emitted in
this prediction (scanned from the real decode prefix, falling back to
`state.section_types` for unit-test convenience). Modal, Auth, and Dashboard
fixtures never emit ImageGallery, so the bias is provably `None` for them
regardless of weight. Four new unit tests in
`tests/test_models/test_compiler_decode.py` cover: firing when the
precondition holds, abstaining without a prior ImageGallery, abstaining for a
different family (Modal) even with a matching missing slot (the structural
incapability proof), and the inert default. All pass.

## Reproducing a checkpoint

The original E620 checkpoint (sha256 `3ce5c9ef...`) that E637/E638 reused is
local-only and was not present on disk or in git history in this session, so
this iteration retrained E620's exact scratch recipe (same train-dir, model,
context backend, output tokenizer, device, steps, batch size, seed) with
run-id `e639-gallery-role-select-checkpoint-20260721`. It completed 800 steps
in 44.76s under `max_wall_minutes=3`, loss 4.068013 — essentially identical
to E620's reported 4.068010. The resulting checkpoint is:

`outputs/runs/e639-gallery-role-select-checkpoint-20260721/checkpoints/last.pt`

SHA-256: `0dc6a4c411e2dbc0d86c4391d8f2f64b6ef1cf3911a91acda8e42131f0ea3927`
(intentionally does not byte-match E620's `3ce5c9ef...`). It is local-only,
not synced, and not promoted.

## Control sanity check: FAILED

The control arm (`gallery_role_select_decode_weight=0.0`) does not reproduce
E637's OOD baseline:

| OOD `n=4` | E637 baseline | E639 control |
| --- | ---: | ---: |
| meaningful v1 | 0.75 | 0.0 |
| strict meaning v2 | 0.5 | 0.0 |
| fidelity | 0.675 | 0.3833 |
| validity | 0.805 | 0.63 |
| structure | 0.581675 | 0.09 |
| component recall | 0.625 | 0.0833 |
| reward | 0.8545 | 0.722 |
| AST node F1 | 0.6437 | 0.1 |
| AST edge F1 | 0.4554 | 0.0 |

Per this session's fresh checkpoint, control's per-record OOD output is:

```
ood_dashboard_01: root = Image(":ood.dash.status.title")
ood_gallery_01:   root = Image(":ood.gallery.img", ":ood.gallery.alt")
ood_modal_01:     root = TextContent(":ood.modal.title", ":ood.modal.body")
ood_auth_01:      root = TextContent(":ood.auth.create")
```

This is qualitatively different from E637's reused-checkpoint output (nested
`Stack`/`ImageGallery`/`Modal`/`Button`/`Input` structures) — the fresh
checkpoint picks a flatter, simpler, and wrong component family (`Image`
instead of the correct nested structures) for every record, not just Gallery.

Investigation before proceeding: the exact `ood`/`remediated`/`n=4`/`cpu`
recipe was confirmed unchanged; the checkpoint sha256 differs as expected
(the original is unrecoverable in this session); three progressively fuller
eval-flag recipes were tried (bare E637 flags; +`--grammar-ltr-primary`
+`--context-backend scratch`; +`--slot-contract-in-context`
+`--semantic-role-decode-weight 8.0`, following the fuller recipe recorded in
E620's own `eval_recipe` block) — all three produced byte-identical,
qualitatively broken predictions, ruling out a missing-flag explanation.
Repeat runs of the same recipe (control r3 vs r4; treatment r1 vs r2 vs r3)
are fully deterministic and byte-identical. The most likely explanation is
CPU floating-point / BLAS-thread nondeterminism compounding over 800
gradient steps on a 1.6M-parameter model, producing a materially different,
weaker local optimum despite an identical seed, train-dir, and near-identical
final loss. This is a checkpoint-reproducibility gap, not an eval-flag or
code-wiring error.

## Treatment result: inconclusive

Because control never selects `ImageGallery` for `ood_gallery_01` in the
first place (it emits `Image(...)` instead), the lever's precondition — an
ImageGallery sibling already open inside the Stack's array — is never
satisfied. Treatment (`gallery_role_select_decode_weight=8.0`) is
byte-identical to control across all 4 OOD records, all metrics, and 3
treatment + 2 control repeat runs. `decode_stats.gallery_role_select_applications`
and `_choice_changes` are both measured-zero (present in
`counters_omitted_zero`, not merely absent) for every treatment run,
confirming the decode-time hook was reached every step but its precondition
never held.

This also means Modal, Auth, and Dashboard are provably untouched in this
run (byte-identical control vs. treatment), consistent with the lever's
structural design — but no positive evidence (Gallery recovering its
sibling) was produced either, because Gallery itself was never reached.

## Decision

Inconclusive, not accepted. The lever is implemented, unit-tested, and
structurally scoped exactly as intended, but this session could not exercise
it on real generation because the freshly retrained checkpoint diverges from
E637's reused checkpoint and never selects ImageGallery at all. No
regression and no improvement were observed — the code is retained
default-off (`gallery_role_select_decode_weight=0.0`), which this session's
own control run confirms is behaviorally inert: `model.twotower`'s v77
decode behavior is unchanged from v76/v75 at default. Re-attempting this
experiment requires either the original E620 checkpoint (unavailable in this
session) or a new scratch checkpoint that reliably selects ImageGallery for
`ood_gallery_01` before the lever's positive-selection behavior can actually
be judged. No checkpoint was synced or promoted.

Evidence: [authoritative treatment JSON](iter-e639-gallery-role-select-20260721.json)
and [control JSON](iter-e639-gallery-role-select-control-20260721.json).
