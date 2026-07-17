# E264 — C2 dynamic pseudo-embeddings for symbol tokens (2026-07-17)

Representation lever + fixture-grade matched row, not a train/ship run. Code:
[`models/twotower.py`](../../src/slm_training/models/twotower.py)
(`_runtime_feature_tensor`, mode `replace`),
[`evals/binding_consistency.py`](../../src/slm_training/evals/binding_consistency.py).
Linear SLM-26.

## What and why

The lexer-native tokenizer's `<SYM_i>`/`<BIND_j>` rows are a fixed learned
pool: slot 0's embedding is whatever training happened to put there, shared
across every example regardless of which placeholder the slot denotes this
time. C2 (DyVo-style dynamic vocabulary; cf. dynamic entity representations,
Ji et al. 2017) replaces the pool row with a **deterministic per-example
pseudo-embedding** of the symbol's referent, so representation tracks
identity, not slot.

## Mechanism

`runtime_symbol_features="replace"` rides the V8 injection path unchanged.
The V8 machinery adds per-example vocabulary-row deltas at both the tied
input embedding and the output projection; replace mode writes
``delta = composed − learned_row`` where ``composed`` is the mean of the
placeholder surface's byte-token embedding rows. Under additive application
the learned pool row cancels exactly:

- input embedding of `<SYM_i>` becomes the byte-compositional vector;
- the tied output logit becomes `hidden · composed`;
- the design constraint holds by construction — no per-example parameters
  (deterministic function of existing tied embedding rows), batching
  untouched (per-example feature tensors, same shapes as V8).

Same surface → identical vector at every slot, position, and example
(test-enforced: exact-row substitution, cross-slot identity, unknown modes
fail loudly).

## Binding-consistency probe

`binding_consistency_probe(model, records)`: mean pairwise cosine of
denoiser hidden states at symbol positions sharing a surface vs differing
surfaces (`binding_margin = same − cross`). With replace mode the *input*
consistency is exact by construction; the probe measures how much survives
the denoiser stack — diagnostic only, no threshold, no gate.

## Recipe

Row E264 (`--matrix v13`): scratch-control train, fixture v1 corpus, 200 CPU
steps, lr 3e-4, seed 0, suites smoke 3 / held_out 5 / adversarial 4 / ood 4 /
rico_held 0 — matched against the recorded E255 control on everything but
`runtime_symbol_features` (registration test enforces the matched-pair
property). `NODE_OPTIONS` overridden (session env poisons the OpenUI bridge).

## Results (fixture-grade, CPU, 2026-07-17)

JSON: [quality-matrix-results-iter-v15-c2-20260717.json](quality-matrix-results-iter-v15-c2-20260717.json);
binding probe: [binding-consistency-e264-20260717.json](binding-consistency-e264-20260717.json).

The first E264 attempt crashed with `runtime symbol feature batch 4 != 5` —
the stale-feature leak PR #275 diagnosed (features set for a training batch
outliving it into a differently-sized eval forward). E264 is the first
merged-lineage row to *activate* runtime features during training, so the
latent defect fired immediately; fixed at the source (`training_loss` clears
features in a `finally`), complementary to #275's loss-suite-entry fix.

| Suite (n) | structural similarity E255 → E264 |
| --- | ---: |
| smoke (3) | 0.300 → 0.189 |
| held_out (5) | 0.323 → 0.261 |
| adversarial (4) | 0.281 → 0.277 |
| ood (4) | 0.372 → 0.290 |

Honest gates fail on both rows (14 thresholds; syntax/meaningful parse 0.0 —
the fixture wall). At this 200-step budget the pseudo-embeddings do **not**
help and mildly depress structural similarity — an honest negative at
fixture scale, consistent with the pool rows having 200 steps of gradient
while composed byte vectors start as untrained averages.

**Binding-consistency probe** (trained E264 checkpoint, 6 surfaces across
fixture records): same-surface hidden cosine **0.9998** vs cross-surface
**0.9679** — binding margin **+0.032**. The exact input-level consistency
replace mode guarantees survives the trained denoiser stack: same symbol →
near-identical representation across positions, separated from other
symbols. That is the metric SLM-26 asked for, now measurable per checkpoint.

## Honesty

Wiring + matched fixture evidence only. No checkpoint promoted, no gate
touched, no ship claim. Whether identity-tracking embeddings lift binding
consistency where it matters (frontier scale, C1 relative refs interacting
with C2 vectors) remains open.
