# SLM-208 (SDE5-01): constraint-debt telemetry fixture (slm208-constraint-debt-20260720)

Matrix set: `slm208_constraint_debt`

Version: `sde5-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

Grammar-mask (legal-token) renormalization distorts the full-vocab distribution in a measurable way; ConstraintDebtV1 exposes the distortion as mass deficit, partition debts, and pre/post-mask KL without changing objectives or gradients.

## Falsifier

The pre/post-mask KL is negative, the legal/full mass invariants are violated, or emitted rows change gradients when telemetry is enabled.

## Three-way interpretation

* **Retain:** The telemetry row schema and KL invariant are internally consistent on synthetic decision events; attach to future preference diagnostic runs.
* **Intervene:** Use real checkpoint/events to verify that the full->legal mass deficit correlates with known grammar-constraint severity.
* **Abandon:** If the KL ever becomes negative or if enabling the writer alters gradient numerics, retract the instrumentation and fix the mass-bundle math.

## Fixture summary

Four synthetic decision events (two decision kinds, train/held-out) evaluated under both full-vocab and legal-renormalized probability spaces. All rows are produced by compute_constraint_debt_v1 directly; no GPU, checkpoint, or model was used.

Total rows emitted: 8 (4 events × 2 probability spaces).

## Aggregate debt by decision kind / split / space

| key | count | legal_debt_mean | good_debt_mean | bad_debt_mean | kl_mean | single_legal_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| component::held_out::full_vocab | 1 | 0.405465 | 1.098612 | 1.098612 | 0.000000 | 0.00 |
| component::held_out::legal_tokens | 1 | -0.000000 | 0.693147 | 0.693147 | 0.405465 | 0.00 |
| component::train::full_vocab | 1 | 0.405465 | 1.098612 | 1.098612 | 0.000000 | 0.00 |
| component::train::legal_tokens | 1 | -0.000000 | 0.693147 | 0.693147 | 0.405465 | 0.00 |
| grammar_comma::held_out::full_vocab | 1 | 1.032584 | 1.440190 | 2.440190 | -0.000000 | 0.00 |
| grammar_comma::held_out::legal_tokens | 1 | -0.000000 | 0.407606 | 1.407606 | 1.032584 | 0.00 |
| grammar_comma::train::full_vocab | 1 | 0.000000 | 0.169846 | 3.169846 | -0.000000 | 0.00 |
| grammar_comma::train::legal_tokens | 1 | -0.000000 | 0.169846 | 3.169846 | 0.000000 | 0.00 |

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The constraint-debt schema, mass/deficit calculations, and pre/post-mask KL are exercised on synthetic events, but no real checkpoint or eval suite was used. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_eval`` until validated with live preference events.

## Honest caveats

- Synthetic logits and events only; no real policy checkpoint or tokenizer.
- No ship-gate claim is made; this is wiring/fixture evidence.
- Pre/post-mask KL is a diagnostic divergence, not a training objective or decoder intervention.
- The small fixture cannot represent tail behavior or real grammar-state heterogeneity.

## Reproducibility

```bash
python -m pytest tests/test_harnesses/preference/test_constraint_debt.py tests/test_harnesses/preference/test_local_train_constraint_debt.py -q
```
