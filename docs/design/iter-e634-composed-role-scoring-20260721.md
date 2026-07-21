# E634 — composed schema-role/repeated-instance scoring at the pre-content boundary

Date: 2026-07-21
Status: completed negative; rejected and reverted; not ship

E633 diagnosed the remaining Auth defect precisely: the schema-opaque
pre-content floor correctly routes the *last* repeated Input instance's
`.name` argument to the legal empty literal, but the repeated-instance margin
independently overrides that same floor for the *first* instance, because the
two scores compete rather than compose. E633's decision named the exact next
step: "compose the two scores at the final choice boundary for only the
pre-content schema pattern." E634 implements that composition and finds it
does not survive contact with the available checkpoint.

## Reused checkpoint and recipe

E620's original rejected local-only checkpoint (SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`), reused by
E631–E633, was not present in this session's gitignored `outputs/runs` (a
concurrent session had left only its own unrelated
`e632-schema-identifier-scratch800-20260721` artifacts there instead). E620's
exact recipe was replayed byte-for-byte — the committed E530 corpus (244
records), scratch context, choice output tokenizer, seed 0, batch size 1, 800
steps, `--no-sync-checkpoints` — completing in 33.29 seconds under the
three-minute cap. The resulting data manifest SHA
(`e65a6ac5a7c49499b638582c325eafd0b245cc4aa9d2650d1396a88230eccee2`) and final
loss (4.068013 vs. E620's 4.068010) match E620 to five significant figures, so
this is a faithful reproduction of the same recipe, not the original artifact.
New checkpoint SHA-256:
`8ab4f5deeb0e1322064259ca92837e71dfdf662d25b46371172ad8779668a4cd`; local-only,
not synced or promoted.

Both baseline and treatment evaluations replayed E631's full OOD `n=4` recipe
(the complete decode-weight set, not just `schema_opaque_decode_weight`) plus
E632/E633's `schema_opaque_decode_weight=4.0`, on this new checkpoint. Both
completed under the three-minute cap with no timeout or fallback and emitted
AgentEvals JSONL plus an AgentV SDK bundle without execution errors.

## Composition attempted

Two variants of `_semantic_plan_repeated_slot_bias`, gated on
`schema_opaque_decode_weight > 0` and a shared `_schema_pre_content_arg` check
(a required non-content string argument directly before a
placeholder-annotated content property — the same guard `_schema_opaque_bias`
already uses):

- **v1 hard gate**: abstain (return `None`) at the pre-content argument,
  deferring entirely to the schema-opaque floor already applied earlier in
  the same decode step.
- **v2 retarget-to-literal**: keep the margin active but point its one-shot
  per-instance floor at the legal empty literal instead of the best unused
  visible slot, at the pre-content argument only.

Both correctly moved the *first* repeated Input instance's `.name` argument
from a visible slot to the literal — the exact defect E633 diagnosed.

## Measured result

| OOD `n=4` | Baseline (reproduced) | v1 hard gate | v2 retarget-to-literal |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.5000 | 0.5000 |
| strict meaning v2 | 0.0000 | 0.0000 | 0.0000 |
| placeholder fidelity | 0.6750 | 0.5083 | 0.5083 |
| placeholder validity | 0.8050 | 0.7050 | 0.7050 |
| structural similarity | 0.5729 | 0.3379 | 0.3379 |
| component recall | 0.6250 | 0.3750 | 0.3750 |
| reward | 0.8515 | 0.7850 | 0.7850 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.3857 / 0.2625 | 0.3857 / 0.2625 |
| latency p50 / p95 | 1316.34 / 5198.77 ms | 1794.83 / 7792.18 ms | 1579.36 / 6593.25 ms |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 |

Auth in the reproduced baseline reproduces E633 exactly:
`v1 = Input(":ood.auth.email", ":ood.auth.name")` (first Input's `.name` still
wrong) and `v4 = Input("", ":ood.auth.email")` (second Input already correct).
Both composed variants change Auth to `root = TextContent(":ood.auth.email")`
— the entire Stack/Button/Input inventory collapses to one unrelated
component. Dashboard, Gallery, and Modal are byte-identical across baseline
and both treatments, confirming the composition engages only on the intended
pattern and is not a broader implementation bug.

## Analysis

The choice-token trace shows the decode is not a fallback substitution —
`syntax_parse_valid` is true and the harness scored the model's own chosen
completion. Forcing the literal at the *first* repeated instance shifts the
greedy decode trajectory early enough in the sequence that this 800-step
scratch checkpoint cannot recover: the trace degenerates into long runs of
structural filler tokens after that point, and decoding settles on a single
unrelated `TextContent` root. Both variants produce byte-identical collapse,
so the failure is insensitive to which scoring formula wins at that argument
(hard gate vs. literal-retargeted margin) — it is about correcting the
family's *first* occurrence rather than its *last*. This reproduces the same
collapse signature E633 r2 hit when it restricted the repeated-slot margin
similarly.

## Decision

Reject both composition variants and revert `model.twotower` to v72
(v71 introduced the composition, v72 records its revert), byte-identical to
the E633-committed v70 code. Do not sync, promote, checkpoint, or claim ship
readiness. The composition idea itself is directionally
confirmed — both variants correctly retarget the first Input's `.name`
argument — but this checkpoint lineage cannot support a correction at the
family's first repeated instance without destabilizing the rest of decode. A
follow-up must either retrain with the corrected first-instance pattern
actually present early in a repeated-family curriculum, or add a decode-time
recovery/verification step that can retry later instances from a stable
state instead of depending on unbroken greedy continuation from the
corrected first argument.

Evidence: [JSON](iter-e634-composed-role-scoring-20260721.json).
