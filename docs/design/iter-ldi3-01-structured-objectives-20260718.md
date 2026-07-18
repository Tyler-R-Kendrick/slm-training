# LDI3-01 (SLM-128) — Legal-Set FTPO, TAB-PO-inspired barrier, TBPO-inspired ratio control

Date: 2026-07-18
Status: **Shared structured-objective library landed with tests and a deterministic
fixture report. No model update, no matrix run, no checkpoint, no quality claim.** The
three objectives are **Adapted** from recent token-level preference work — not
reproductions or SOTA claims.

## Why this exists

The local-objective library had `ce_margin` / `unlikelihood` / `ftpo_single` / `ftpo_set`
in `local_train.py`. LDI3-01 adds three OpenUI-native structured objectives over the
exact grammar-legal action sets that both the causal and TwoTower trainers can share,
building on the DecisionEventV2 materializers and objective-support admission from SLM-116
/ SLM-117 (both on `main`).

## What landed

`src/slm_training/harnesses/preference/structured_objectives.py` — one architecture-neutral
implementation (pure functions of a 1-D logit row + materialized action sets; the trainer
extracts the row, this module owns the math):

- **`StructuredObjectiveConfig`** — typed, versioned, validated, fail-closed-on-unknown-fields
  config with a deterministic `content_sha` fingerprint (part of run/adapter identity).
- **`StructuredObjectiveInput`** — the exact decision: logits + legal / good / bad /
  ambiguous / unobserved sets + per-action evidence weights, semantic roles, criticality,
  and an optional same-state reference. Rejects empty good/bad sets, out-of-legal ids, and
  overlapping partitions.
- **Objective A — Legal-Set FTPO**, two variants:
  - *pairwise margin*: explicitly pair-weighted `G × B` set margin
    `weighted_mean(evidence · clamp((ε−δ)/ε) · softplus((ε−δ)/τ))`, normalized per state;
  - *mass margin*: `softplus(margin + log P_B − log P_G)` over the legal-token softmax,
    reporting good / bad / ambiguous / unobserved mass separately.
- **Objective B — TAB-PO-inspired barrier**: additive, separately-metered SFT anchor
  `−log p_legal(g)` for good actions that are verifier-critical *and* under `barrier_p`,
  weighted by evidence × semantic role; zero for confident good actions; structural roles
  default to low weight. `token_erosion_rate` reports the TAB failure mode (good likelihood
  falls while relative preference rises).
- **Objective C — TBPO-inspired ratio control**: bounded good-vs-bad log-ratio control
  against the same-state reference, advantage-centered or centered by a small serializable
  `StateBaseline` fit strictly from train states; disabled and reported when there is no
  reference (inadequate state support).
- **Composition**: `structured_objective_loss` composes preference + optional barrier +
  the target / non-target MSE locality tethers, each **separately metered** so the barrier
  and target tether never double-count. `structured_objective_batch_loss` averages
  per-state losses so large action sets cannot dominate (with an un-normalized
  pair-weighted mode kept for ablation). `structured_objective_report` is the no-update
  evidence generator.

Existing `ce_margin` / `unlikelihood` / `ftpo_*` are untouched — the new objectives carry
new names, so historical behavior and tests are unchanged (68 existing preference tests
green).

## Fixture demonstration (wiring evidence only)

[`iter-ldi3-01-structured-objectives-fixture-20260718.json`](iter-ldi3-01-structured-objectives-fixture-20260718.json)
runs every objective over a shared 3-state fixed-logit corpus (6-token vocab, 4-token
legal set). It is fixture evidence only — fixed logits, **no model update**.

- The objectives are genuinely distinct on the same corpus: batch loss ≈ mass-margin
  `1.26`, pairwise `2.73`, barrier `2.36`, ratio-control `1.26`.
- **Raw vs legal space differs**: state 1 good mass is `0.39` in full-vocabulary softmax
  vs `0.59` in the legal-token space (the two illegal tokens carry raw mass that the
  constrained objective correctly excludes).
- **Barrier is selective**: only the state whose critical good action sits below
  `barrier_p` activates the anchor (`[0, 0, 1]` across the three states).
- **Erosion** is detected on a before/after example (a good token's absolute probability
  falls while relative preference improves).

## Acceptance coverage

`tests/test_harnesses/preference/test_structured_objectives.py` (16, green): both
Legal-Set FTPO variants match hand-computed values; state normalization prevents
large-set dominance; legal mass sums over only the legal set (a `100`-logit illegal token
contributes nothing); barrier activates only for critical, under-confident good actions and
respects structural-role weighting; erosion detection; ambiguous/unobserved are normalized
but never targets; ratio control compares at identical states and disables without a
reference; the state baseline fits from train and round-trips; mass-margin gradients match
finite differences; tiny-probability numeric stability; config round-trip + deterministic
fingerprint + fail-closed validation; the same implementation is architecture-neutral over
"causal" and "TwoTower" mock logits; and input validation rejects illegal/empty/overlapping
sets. `ruff` and `python -m scripts.repo_policy` clean.

## Honesty

**Adapted, not reproduced.** These are independent adaptations of the documented mechanisms
in TAB-PO ([arXiv:2603.00025](https://arxiv.org/abs/2603.00025)) and TokenRatio/TBPO
([arXiv:2605.12288](https://arxiv.org/abs/2605.12288)); the names carry `tab_po_inspired` /
`tbpo_inspired` precisely because the paper equations are not faithfully reproduced. No
quality-bearing matrix run, no new event mining, no trainer-specific duplicate
implementation, and no hidden eval gold. Criticality comes from compiler/AST role and
independently verified action evidence, never gold final programs.

## Honest remaining scope

- Wiring the shared objectives into the causal and TwoTower training entry points behind a
  config switch (a training change, deferred; the objectives are the single implementation
  those adapters would call).
- Confusion-aware negative families from empirical action tables (the barrier supports
  evidence weights; the empirical-table sourcing is follow-on).
- The learned-scalar baseline is a mean-log-ratio fit; a capacity-bearing baseline is a
  later option if state normalization proves worth it.
