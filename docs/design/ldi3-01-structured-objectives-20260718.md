# LDI3-01 — Structured local-preference objectives

**Issue:** SLM-128 (LDI3-01). **Date:** 2026-07-18. **Status:** implemented +
gradcheck-verified. **No model update or quality claim** — objectives are
*adapted*, not reproduced, from the cited work.

## What this delivers

A shared, architecture-neutral objective library consumed by both the causal and
TwoTower local trainers (one implementation, no per-architecture duplicate math):

- `src/slm_training/harnesses/preference/structured_objectives.py`
- `scripts/report_structured_objectives.py` — no-model-update objective report
- `tests/test_harnesses/preference/test_structured_objectives.py` — 14 tests
- `docs/design/ldi3-01-structured-objective-report-20260718.json` — committed
  fixture report

`structured_decision_loss(logits, view, *, legal_action_ids, config, ...)` takes
1-D decision logits + a materialized `ObjectiveView` and returns `(loss, detached
metrics)`. Because it only needs 1-D logits, the *same* call serves causal
(next-token) and TwoTower (canvas) logits.

## Objectives

**A — Legal-Set FTPO** (`legal_set_ftpo`), two variants:
- `pairwise`: weighted margin over verified `G × B` pairs,
  `weight = evidence · clamp((ε−δ)/ε, 0, 1)`, `softplus((ε−δ)/τ)`, **per-state
  mean** so large action sets do not dominate the batch.
- `mass`: `softplus(margin + log P_B − log P_G)` over the legal-softmax masses —
  the primary constrained-space objective; masses sum over the declared legal set
  only.

**B — `tab_barrier`** (TAB-PO-inspired): the pairwise preference term **plus** an
additive, separately-metered SFT anchor `−log p(g)` applied *only* to verified,
critical (`critical_good_mask`), under-confident (`legal prob < barrier_p`) good
actions — zero for confident ones. Reports `barrier_active_fraction`,
`barrier_loss`, and a likelihood-**erosion** rate vs a reference.

**C — `tbpo_inspired`** (TokenRatio/TBPO-inspired): compares good/bad log-ratios
against a reference at the *same* state, `softplus(margin − (r̄_G − r̄_B))`, with
an optional advantage-centered `state_baseline`. Named `_inspired` (not `TBPO`)
and disabled when legal-state support is inadequate.

## Guarantees (tested)

- Both FTPO variants match hand-computed toy values; legal mass sums over the
  legal set only (an illegal high-logit action is excluded).
- **`torch.autograd.gradcheck` passes** for all four objective/variant forms.
- Numeric stability at vanishing masses (finite loss and gradients).
- Barrier activates only for under-confident critical good actions; erosion is
  reported vs a reference.
- Config **fails closed** on unknown fields and invalid ranges; fingerprint
  round-trips and is part of run/adapter identity.
- Non-trainable (constraint-shadow) views and out-of-legal action ids are refused
  (the E284 lesson: legality is not a semantic label).
- Existing `unlikelihood`/`ftpo_*` behavior in `causal_local_train`/`local_train`
  is untouched — this is an additive shared library.

## Commands

```bash
python -m pytest tests/test_harnesses/preference/test_structured_objectives.py -q   # 14 passed
python -m ruff check src/slm_training/harnesses/preference tests/test_harnesses/preference
PYTHONPATH=src python -m scripts.report_structured_objectives \
    --out docs/design/ldi3-01-structured-objective-report-20260718.json
python -m scripts.repo_policy
```

## Scope

This implements the objectives, config, gradient correctness, and the fixture
report. Wiring the new names into the causal/TwoTower trainer dispatch for a
quality-bearing run is deferred to the LDI3 matrix issues (out of scope here:
"No quality-bearing matrix run").
