# Thrash timing Pareto calibration (2026-08-03)

## Product rule

```text
timeout ≈ optimal_target_runtime + variance_margin(error_ratio)
```

| Incomplete / timeout hit rate | Interpretation | Action |
| --- | --- | --- |
| **High** (≫15%) | Budget/recipe calc too tight | Recalculate (prefer shrink recipe; raise L0 only if recipe at floor) |
| **Low** (≪5%) with large slack | Budget too loose | Tighten for more experiments/day |
| **~5–15%** | Near equilibrium | Hold |

Never ad-hoc wall++ because a cycle failed. Never silent widen mid-campaign.

## Failure that forced this

Under `MAX_RUN_MINUTES=3`, continuous arm share is ~**70s**.  
Screening used **decode 24s × smoke n=3 ≥ 72s** → eval alone could exceed the arm wall → near-100% `empty_metrics` / incomplete thrash.

## Immediate recalibration (locked in policy)

| Knob | Before | After | Rationale |
| --- | ---: | ---: | --- |
| `screening_decode_timeout_seconds` | 24 | **8** | 3×8=24s eval budget; leaves train floor |
| `thrash_timing.screening_thrash_steps` | (80 CLI) | **40** | Micro-train fits remaining arm share |
| `promotion_decode_timeout_seconds` | 24 | 24 | Promote still dual-suite; rare |

Fit helper: `_fit_screening_decode_timeout_seconds` clamps further if policy decode would exceed  
`arm_wall − min_train_floor − eval_overhead` / n.

## Telemetry

Each cycle writes `thrash_timing.json` + appends `loops/<loop>/thrash_timing.jsonl`:

- `complete`, `measurement_complete`, `has_dual_arm_ss`
- `arm_wall_seconds`, `decode_fit`, `incomplete_reasons`

Use incomplete rate + p95 arm times to recompute the thrash_timing policy (version bump), not vibes.

## Follow-up fixes (executable thrash)

After decode/steps fit, residual incomplete causes were **not** wall:

1. **`semantic-contrast` train invariant:** `batch_size >= 3` required; thrash defaulted to 2 → every contrast arm failed train. Bank extras + knobs() now force `batch_size >= 3` when contrast loss > 0.
2. **Agent hypothesize feedback conflict:** thrash matrices carried stale `feedback_ids` from distant ancestors while live lineage feedback was empty → `agent hypothesis matrix conflicts with supplied feedback ids`. Driver strips orphan feedback bindings when no live feedback exists.

## Non-goals

- Raising `MAX_RUN_MINUTES` as default thrash fix  
- Regime-epoch arm recycle  
- Counting incomplete cycles as thrash evidence  
