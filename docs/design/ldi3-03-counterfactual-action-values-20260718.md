# LDI3-03 — Verifier-backed counterfactual action-value probe

**Issue:** SLM-131 (LDI3-03). **Date:** 2026-07-18. **Status:** probe orchestration
+ value materializers, implemented and tested. **No model rollout, no verifier
run, no training** — the model-dependent surface is deferred behind a backend
protocol.

## What this delivers

Orchestration that turns delayed OpenUI failures into defensible *same-state*
action evidence by forcing compiler-legal alternatives at the exact stored state,
rolling each forward under common seeds, and deriving pure value materializers:

- `src/slm_training/harnesses/preference/counterfactual_probe.py`
- `scripts/run_counterfactual_probe.py` — bounded mock-backed evidence report
- `tests/test_harnesses/preference/test_counterfactual_probe.py` — 7 tests
- `docs/design/ldi3-03-counterfactual-probe-report-20260718.json` — committed fixture

The only model-dependent surface is `RolloutBackend.rollout(state, action, seed)`
("the model plug-in only forces an action and produces an outcome"). Everything
else is pure and deterministic.

## Honesty-critical rules (tested)

- **Admission**: `heuristic_only` (final-output-blame) candidates are quarantined
  — rejected unless explicitly requested, and never admitted to the semantic
  queue. Delayed-failure blame is never inferred from final-output position; only
  from forced-action rollout verdicts.
- **Action selection**: from the exact `legal_action_ids` (never global token
  top-k); always includes the policy action; all legal actions within a cap, else
  deterministic by policy prob → role coverage → id, recording the excluded set.
- **Identity caching**: keyed by `(state, action, seed, policy_sha, decoder_hash,
  verifier_hash)`, so a changed policy/decoder/verifier identity cannot reuse
  incompatible outcomes; a populated cache makes a run resumable to the same
  manifest.
- **Value/verdict materializers**: Pareto front, lexicographic key, scalar reward,
  and a binary verdict that returns **`unresolved`** when a required gate is not
  observed in every rollout — never a silent pass/fail.
- **Semantic partition**: good/bad measured against the *policy action* baseline,
  requiring ≥ `min_rollouts`, a `min_effect` margin, and no required-hard-gate
  regression; thin evidence → ambiguous; a legal action never probed → unobserved.
  The full ordered G0-G12 verifier vector is retained per outcome
  (`ActionOutcomeV2`).

## Commands

```bash
python -m pytest tests/test_harnesses/preference/test_counterfactual_probe.py -q  # 7 passed
python -m ruff check src/slm_training/harnesses/preference tests/test_harnesses/preference
PYTHONPATH=src python -m scripts.run_counterfactual_probe \
    --out docs/design/ldi3-03-counterfactual-probe-report-20260718.json
python -m scripts.repo_policy
```

The fixture report yields 2 good, 1 bad, 1 ambiguous (the policy baseline), and 1
unobserved action — the required multi-partition fixture.

## Scope

Admission, selection, identity-caching, common-seed orchestration, resume/dedup,
and the pure value/verdict/partition derivations. The ship-grade run swaps the
mock backend for a real model + the G0-G12 verifier stack + independent judge
(compute-bound, out of scope here). No model training; partial/timeout evidence
stays unresolved and fails strict corpus admission.
