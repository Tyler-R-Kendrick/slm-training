# AP-028 adaptive-router admission disposition

SLM-323 requires an adaptive plan-length/refinement router trained only on
development or cross-validation outcomes from AP-027. The current AP-027
artifact does not admit that training.

## Audit (2026-07-26)

`iter-slm322-discrete-plan-pareto-20260726.json` contains two identical,
single-seed, round-one fixture rows for one record. Its quality fields are
explicit parse-rate and placeholder-fidelity proxies; rounds 2/4/8, matched
arms, per-example outcomes, p95/compute risk, and split/group identifiers are
absent. It cannot support a learned policy, calibration claim, or a fixed-policy
comparison.

The available target-derived features are inadmissible: semantic bits,
AST/component/binder counts, and plan features read the target OpenUI program
or placeholders. They would leak locked labels. The evaluator's temporal decode
evidence is the correct seam: it preserves prefix-time position, legal-set,
forced, phase, decision-source, and choice-change data while keeping final
parse/semantic/error/timeout/fallback outcomes as labels.

## Disposition

**Adaptive-router integration is unavailable; no router, training, checkpoint,
or metric curve was added.** No local or remote train/eval, human-rating gate,
or ship claim is made.

A successor run must first provide a locked AP-027 development protocol with:

1. matched per-example 1/2/4/8-round outcomes across fixed and candidate arms;
2. explicit prompt/prefix-time-only features, with target-derived fields
   rejected and all decisions logged;
3. group-disjoint development, frozen calibration, and locked-test splits by
   target cluster, checkpoint, arm, and seed;
4. frozen calibrator hash, ECE/Brier, abstention and compute-risk accounting;
   and a logged conservative fixed-policy fallback with its full cost.

Until then, AP-027's fixture proxies remain wiring evidence only and cannot
justify adaptive routing or unblock an AP-028 quality/latency claim.
