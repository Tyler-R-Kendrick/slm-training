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
2. **Agent hypothesize feedback conflict (empty lineage):** thrash matrices carried stale `feedback_ids` while live lineage feedback was empty → strip orphan binds.
3. **Agent hypothesize feedback conflict (handoff ≠ lineage):** continuous `pred` is the last **handoff** campaign; hypothesize walks **full loop lineage** and may bind a different formed matrix (e.g. incomplete next cycle with partial feedback). Pinning handoff `feedback_ids` into thrash matrices aborts with `agent hypothesis matrix conflicts with supplied feedback ids` even when both sides are non-empty. **Thrash never binds predecessor feedback** — only confirm/promote/replay do. `AgentHypothesisProvider` rebinds from live lineage.

## Measured-cost consumer (2026-09-02, `harness.autoresearch.experiment_campaign` v272)

The rule above was policy-only until v272: `incomplete_rate_high/low` had no
consumer, and `_fit_screening_decode_timeout_seconds` clamped the 12 s constant
to the wall from an assumed 2 s floor while the measured per-record decode cost
on this box class was 23–35 s (`continuous-openui-scheduled-fe71636-c4-results.json`,
`-gmyilq-c3-results.json`). The fit now reads the predecessor's
`runs/*/eval_smoke.json` (`details[].latency_ms`, falling back to
`latency_ms_p95_including_incomplete` / `latency_ms_p95` / `compiler_ms_mean`):

- `decode_floor_source` = `measured_p95` (nearest-rank p95 over decoded records,
  censored walls counted as observed) or `policy_default`
  (`screening_sample_size.default_decode_floor_seconds`, a fallback only). The
  floor is passed to `screening_smoke_n_for_policy(per_record_decode_floor_seconds=)`.
- `n_probe = max(1, floor(eval_share / p95))`,
  `fitted_decode_timeout_seconds = min(eval_share / n_probe, p95 × (1 + p95_margin))`
  with `eval_share = arm_wall − min_train_floor − eval_overhead`. Never wall++;
  `exceeds_configured` records when the policy constant was infeasible.
- Pareto consumer (`pareto.decision` / `pareto.reason`): predecessor incomplete
  rate `> incomplete_rate_high` → `shrink` (floor inflated by the margin, so
  `n_probe` drops; `shrink_steps` when one probe is train-bound and the eval
  share is kept for decode instead of growing the train floor);
  `< incomplete_rate_low` → `grow`; in band → `hold`; no measurement → `cold_start`.
- `timeout_cause` = `budget_timeout` (p95 > applied timeout: the budget was
  infeasible) | `slow_decode_timeout` (feasible budget, slow tail) | `none`.
  A `budget_timeout` never sets `compiler_ms_timeout` and never routes into
  `DECODE_RESIDUAL_SLUGS` (measured effect of those arms under budget
  timeouts: 0.0 over 60 observations) — it is recalibrated by this fit.

`thrash_timing.json` lifts `decode_floor_source`, `decode_floor_seconds`,
`n_probe`, `fitted_decode_timeout_seconds`, `pareto`, `timeout_cause` beside the
existing keys.

## Non-goals

- Raising `MAX_RUN_MINUTES` as default thrash fix  
- Regime-epoch arm recycle  
- Counting incomplete cycles as thrash evidence  
