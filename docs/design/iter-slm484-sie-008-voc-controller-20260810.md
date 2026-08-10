# SLM-484 (SIE-008): Value-of-compute controller campaign (EXP-SR-5)

**Claim class:** `fixture` only (catalogue identity claim_class=`diagnostic`; execution is fixture — no promotion)

**Catalogue:** `exp-sr-5`

**Primary metric (`compute_value_regret`, symbolic arm mean):** 0.503562

**Effect vs hand-threshold (control − symbolic):** `-0.290599` (minimum_effect=`0.05`)

**Ceiling recovery vs gold oracle:** `0.0`

**Recommendation:** `reject` — Catalogue falsifier holds or safety audit failed; controller stays off.

## Acceptance snapshot

| Check | Value |
| --- | --- |
| legal_support_parity_exact | True |
| falsifier_holds | True |
| clears_hand_threshold_control | False |
| clears_minimum_effect | False |
| paired Δ vs hand-threshold (mean ± SE) | -0.290598 ± 0.089879 (n=3) |
| seeds | [0, 1, 2] |
| promotion | False |

## Arms (mean regret across seeds; lower is better)

| Arm | compute_value_regret_mean | per-seed |
| --- | ---: | --- |
| hand_threshold | 0.212963 | [0.166667, 0.166667, 0.305556] |
| linear_scorer | 0.074074 | [0.138889, 0.055556, 0.027778] |
| symbolic_rule | 0.503562 | [0.555556, 0.538462, 0.416667] |
| gold_oracle | 0.0 | [0.0, 0.0, 0.0] |

## Falsifier / safety notes

- Catalogue falsifier: symbolic regret not below hand-threshold control by minimum_effect on eval traces.
- Controller recommends among authorized compute strategies only; legal domain is grammar-owned and unchanged (I6).

## Scope

Fixture-scale EXP-SR-5 value-of-compute controller replay over deterministic synthetic DecodeStatsRecordV1 traces. Arms compose SIE-007 compute_state_controller (hand threshold, linear scorer fit on discovery only, preregistered symbolic IR, gold oracle ceiling). Primary metric is compute_value_regret on held-out eval traces (decrease is better). Discovery/eval are disjoint. I2 singleton bypass and I6 legality parity are audited. claim_class=fixture; no promotion or production default change.

Command: `python -m scripts.run_sie008_voc_controller --mode fixture`

Full detail: `docs/design/iter-slm484-sie-008-voc-controller-20260810.json`.
