# Screening power analysis (paired smoke eval)

**Claim class:** diagnostic / fixture — not a ship claim. Gates unchanged.

**JSON sidecar:** [screening-power-analysis.json](screening-power-analysis.json)

## Recipe

| Field | Value |
| --- | --- |
| Host / device | local CPU (artifacts from `continuous-openui-local` 20260820) |
| Source | `outputs/autoresearch/continuous-loop-20260820-*/runs/*/eval_smoke.json` |
| Metric | `smoke.structural_similarity` per-record `details[]` |
| Pairing | same fixture id, control vs candidate arm |
| Control files | 144 |
| Paired cycles | 109 |
| n deltas | 405 |
| Tie share | 0.316 |
| Alpha / power | 1/20 / 0.8 (paired z-power, `autoresearch.power`) |
| Eval backend | scratch, grammar-constrained |
| Source suite n | 6 (`e938_role_safe_all_targets_smoke6_v1`, frozen) |

## Variance components

Cross-cycle control means mixed fixture, seed, and knob-bleed. Paired deltas
are the screening statistic.

| Component | Estimate |
| --- | ---: |
| Pooled per-record σ | 0.119 |
| Mean per-id σ | 0.100 |
| Between-cycle σ of suite mean | 0.077 |
| **Paired-delta σ** (used for MDE) | **0.1741** |
| σ of cycle mean delta | 0.131 |

## MDE at candidate n (paired)

| n | MDE (80% power, α=0.05) |
| ---: | ---: |
| 6 | 0.199 |
| 12 | 0.141 |
| 24 | 0.100 |
| 48 | 0.070 |
| 21 (arm-wall ceiling at 2s/record) | 0.106 |

`required_n_for_effect(0.01, σ=0.1741)` = **2380**. Policy `minimum_effect:
0.01` is not detectable at any n the arm wall can buy. Do not lower α or
gates; raise the declared effect to the measured MDE or buy more n.

## Certified screening n

- Decidability floor (sign test, α=1/20): **6**
- Power floor at honest MDE≈0.106: **21**
- Suite volume after this change: **24** (`e938_role_safe_all_targets_smoke24_v1`)
- Budget ceiling (70s wall − 20s train − 8s overhead, 2s/record): **21**
- **Chosen screening n (feasible):** `max(6, 21)` clamped by ceilings = **21**
- Frozen `e938_*` snapshots were not mutated.

`promotion_suite_n` interface request: **24** (suite volume; promotion timeout
is 24s/record so the confirm tier can spend the extra three records). Pair
with `minimum_effect ≈ 0.106` (or 0.10) on the screening primary. Leaving
`minimum_effect: 0.01` keeps `power_floor_n=2380` and `must_generate=true`.

## Data quality (eval build)

`python -m scripts.build_test_data --source fixture --version
e938_role_safe_all_targets_smoke24_v1` against `wf_smoke_v2`: smoke n=24,
`leakage_rejected=0`, `error_count=0`. Recheck vs `e937_role_safe_all_targets_v2`
also disjoint. `smoke_image_01` was rewritten (Image+Callout) after the first
build rejected Image+Text+Button as `openui+openui_structure` overlap with
`train_image_01_aug_cta`. No gate was weakened.

## Honesty

This is a **fixture-scale diagnostic**. It does not authorize ship, promotion,
or `rico_held` claims.
