# SLM-212 (SDE5-05): constraint-debt routing fixture (slm212-debt-routing-20260721)

Matrix set: `slm212_debt_routing`

Version: `sde5-05-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

A deterministic constraint-debt router can choose the cheaper MaskGIT path when legal-mass debt is low and the stricter constrained-LTR path when debt is high, improving the quality/latency frontier over fixed policies at matched verifier budgets.

## Falsifier

The static or calibrated router does not match or exceed the better fixed policy after budget matching; the signal-permuted control achieves the same regret; debt is not calibrated enough to choose the winning decode path; or gains require unequal verifier/forward budgets.

## Arms

| arm_name | accuracy | mean_outcome | mean_regret | total_verifier_cost | route_counts |
| --- | --- | --- | --- | --- | --- |
| fixed_maskgit | 0.4333 | 0.6673 | 0.1454 | 30.0 | {'maskgit': 30} |
| fixed_ltr | 0.3667 | 0.6407 | 0.1721 | 90.0 | {'ltr': 30} |
| fixed_asap | 0.2000 | 0.6174 | 0.1953 | 45.0 | {'asap': 30} |
| static_debt_router | 0.7667 | 0.7468 | 0.0659 | 54.0 | {'ltr': 12, 'maskgit': 18} |
| calibrated_debt_router | 0.7667 | 0.7468 | 0.0659 | 54.0 | {'ltr': 12, 'maskgit': 18} |
| signal_permuted_router | 0.4667 | 0.6582 | 0.1545 | 54.0 | {'maskgit': 18, 'ltr': 12} |
| oracle_router_ceiling | 1.0000 | 0.8127 | 0.0000 | 55.0 | {'ltr': 11, 'asap': 6, 'maskgit': 13} |

## Signal and thresholds

- Signal: `D_legal`
- High threshold: 2.0
- Low threshold: 2.0
- Hysteresis: 1
- Budget mode: `equal_verifier_budget`

## Disposition

**signal_predictive**

Static debt router matches or beats the best fixed policy and improves over the signal-permuted control, suggesting the signal carries real routing information in this synthetic fixture.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The router policy, calibrator fallback, hysteresis, and budget accounting are exercised over deterministic synthetic states, but no real model or decode path was run. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until trained-model constraint-debt telemetry and AgentV evaluation are available.

## Honest caveats

- Synthetic fixture: signals and outcomes are randomly generated and only weakly correlated with the true-best route; real constraint-debt telemetry will differ.
- No model, checkpoint, GPU, or verifier labels were used; this is wiring evidence only.
- The oracle router uses synthetic outcome scores, not serving-time signals, and is a diagnostic ceiling only.
- Budget accounting is a synthetic verifier-cost proxy, not measured wall time or forward passes on a real checkpoint.
- No ship-gate claim is made; the route ceiling and signal calibration must be re-evaluated with real decode paths and AgentV evaluation.

## Reproducibility

```bash
python -m scripts.run_slm212_debt_routing_fixture --mode plan-only
python -m scripts.run_slm212_debt_routing_fixture --mode fixture
```
