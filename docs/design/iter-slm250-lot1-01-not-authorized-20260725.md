# SLM-250 LOT1-01 — not_authorized disposition

## What

LOT1-01 asks to implement the faithful causal K×c looped-latent model path,
gated on two hard activation gates against its own upstream contracts:

1. SLM-248 (LOT0-01) `LotusOpenUIFidelityContractV1.authorization.verdict`
   must be `authorize_bounded_implementation`.
2. SLM-249 (LOT0-02) `CompilerReasoningTraceGateV1.gate.verdict` must be
   `oracle_ceiling_positive` (or another explicit authorization supplying
   K/c/stage targets).

"Otherwise close `not_authorized` in plan-only mode without production model
code."

## Evaluated result

Both gates were evaluated against the real, committed upstream artifacts:

- **Gate 1** (`docs/design/lotus-openui-fidelity-contract-v1.json`): actual
  verdict is `needs_target_trace_contract`, not `authorize_bounded_implementation`.
  **Unmet.**
- **Gate 2** (`docs/design/compiler-reasoning-trace-v1.json`): actual verdict
  is `inconclusive`, not `oracle_ceiling_positive`; the gate's own
  `allowed_lot1_implementation` field explicitly says
  `"none: ... not authorized by this issue"`. **Unmet.**

Both gates unmet ⇒ **verdict: `not_authorized`**. No K×c latent-workspace
model, loop driver, curriculum hooks, or training code is added.

## What was built instead

Since evaluating this gate honestly *is* the LOT1-01 deliverable when the
gate is unmet, this issue adds a small, reusable, tested activation-gate
evaluator rather than a bare Linear comment:

- `src/slm_training/harnesses/experiments/lot1_01_activation_gate.py` —
  `LotusOpenUIModelContractV1` schema, `GateEvaluation`, and
  `evaluate_activation_gates()`, a pure function that reads the two upstream
  contract dicts and derives the verdict from their real published fields
  (never hardcoded to always fail — synthetic contracts reporting both
  required verdicts flip the result to `authorized_wiring_only`, tested).
- `scripts/evaluate_lot1_01_activation_gate.py` — plan-only CLI; loads the
  two real committed JSON artifacts and emits the disposition. No model
  import, no training, no GPU path.
- Tests proving: the real current contracts yield `not_authorized`; a
  synthetic both-gates-met case yields `authorized_wiring_only`; a
  mixed/missing-fields case fails closed to `not_authorized`; the contract
  hash is stable and changes with the verdict.

This evaluator is reusable: once SLM-249 is rerun with a real oracle-ceiling
campaign (or SLM-248's verdict changes), re-running the same CLI against the
updated artifacts will honestly reflect the new disposition without any
narrative rewrite.

## Files added

- `src/slm_training/harnesses/experiments/lot1_01_activation_gate.py`
- `scripts/evaluate_lot1_01_activation_gate.py`
- `tests/test_harnesses/experiments/test_lot1_01_activation_gate.py`
- `tests/test_scripts/test_evaluate_lot1_01_activation_gate.py`
- `docs/design/iter-slm250-lot1-01-not-authorized-20260725.md`
- `docs/design/iter-slm250-lot1-01-not-authorized-20260725.json`

## Commands

```bash
python -m scripts.evaluate_lot1_01_activation_gate \
  --fidelity-contract docs/design/lotus-openui-fidelity-contract-v1.json \
  --trace-gate-contract docs/design/compiler-reasoning-trace-v1.json \
  --out outputs/runs/slm250_activation_gate
```

## Verification

- `pytest tests/test_harnesses/experiments/test_lot1_01_activation_gate.py tests/test_scripts/test_evaluate_lot1_01_activation_gate.py -q` → 9 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Acceptance criteria mapping

- "close `not_authorized` in plan-only mode without production model code" —
  satisfied exactly; no model/training file is touched.
- "no RSC/TwoTower path is reused as the treatment" — no model path is
  touched at all.
- "no production default change" — satisfied.
- LOT1-02/LOT2/LOT3/LOT4 remain gated behind LOT1-01, unchanged.

## Non-goals honored

No K×c latent-workspace model code, no learned plan predictor, no causal
intervention campaign, no large training run or quality claim, no adaptive
depth/halting, no production default change.
