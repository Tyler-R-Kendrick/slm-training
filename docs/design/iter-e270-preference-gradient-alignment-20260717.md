# E270 — preference gradient split alignment

Date: 2026-07-17
Status: **in progress; diagonal profile requires cross-kind matrix**

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
