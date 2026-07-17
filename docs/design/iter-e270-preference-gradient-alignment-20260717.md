# E270 — preference gradient split alignment

Date: 2026-07-17
Status: **completed; raw MGDA direction aligns held-out losses; optimizer geometry is the mismatch**

E270 profiles the frozen E228 parent without optimizer or checkpoint mutation.
It computes exact `ftpo_set` gradients for committed E261 events, grouped by
grammar/AST `decision_kind` and split, together with event/group/evidence
provenance.

The initial profile fetched/rebased latest `origin/main`, was clean, and proved
`0 behind / 1 ahead` at `4f686f5`. All 12 decision kinds shared by train and
held-out have nonnegative same-kind gradient cosine. Ten are strongly aligned
(`0.56` to `0.999`), `bind_reference_bound_children` is weakly positive
(`0.105`), and `bind_declaration_root` is inactive in both splits. Two train
kinds have no held-out counterpart: `bind_reference_bound_input` and
`grammar_lsqb`.

This rules out a simple same-kind train/held sign reversal, but it does not
explain E269: a train gradient for one decision kind can oppose a held-out
gradient for another. The final profile therefore adds the full held-kind by
train-kind alignment matrix and each held gradient's alignment to the MGDA
train combination before drawing a data conclusion.

Initial diagonal evidence:
[`quality-matrix-v10-e270-gradient-alignment-diagonal-results.json`](quality-matrix-v10-e270-gradient-alignment-diagonal-results.json).

## Cross-kind result

The final profile repeated the clean/rebased gate (`0 behind / 2 ahead`) and
emitted the full held-kind by train-kind matrix. Cross-kind interference is
real and often severe. The strongest negative cosine pairs include:

| Held-out kind | Train kind | Cosine |
| --- | --- | ---: |
| `grammar_comma` | `grammar_rsqb_bound_populated` | -0.9941 |
| `grammar_comma` | `grammar_rsqb_root_populated` | -0.9929 |
| `grammar_rsqb_bound_populated` | `grammar_comma` | -0.9127 |
| `grammar_rsqb_root_populated` | `grammar_comma` | -0.9120 |
| `grammar_rpar` | `grammar_comma` | -0.8393 |
| `component_root` | `component_bound` | -0.5392 |

MGDA nevertheless combines the train gradients so that every active held-out
FTPO-loss gradient has positive dot product with the raw combined direction.
The weakest active held-out alignments are `grammar_comma` (cosine `0.0032`),
`grammar_rsqb_bound_empty` (`0.0159`), and
`grammar_rsqb_bound_populated` (`0.0167`); all remain nonnegative.

## Decision

The E269 failure is not explained by train/held same-kind reversal or by the
raw MGDA combination opposing held-out FTPO losses. The remaining mismatch is
optimizer geometry: the certificate covers the raw combined gradient, while a
first AdamW step applies adaptive/sign-like preconditioning plus decoupled
weight decay. E269's finite optimizer step therefore need not preserve the raw
gradient dot products.

Retain the committed judged events. Before any further training, diagnose and
certify the actual optimizer-transformed proposal direction against train and
held-out per-kind objectives, or use an optimizer whose update is collinear
with the certified direction. Do not synthesize case-specific data or tune a
scalar learning rate based on this result.

Final machine-readable evidence:
[`quality-matrix-v10-e270-gradient-alignment-results.json`](quality-matrix-v10-e270-gradient-alignment-results.json).
