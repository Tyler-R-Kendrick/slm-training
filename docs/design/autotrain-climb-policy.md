# Autotrain climb policy (externalized)

**Status:** committed versioned policy + loaders; continuous Phase A and
loop-scoped exhaust use it. Live multi-cycle GPU trains are out of scope for
this note.

## Owner artifact

| Field | Value |
| --- | --- |
| Path | `src/slm_training/resources/experiments/autotrain_climb/policy.v1.json` |
| Schema | `autotrain_climb_policy/v1` |
| Loader | `slm_training.autoresearch.climb_policy.load_climb_policy` |

High-volatility continuous/climb knobs live in that JSON (primaries, cadence,
exhausted-identity fields, recipe-null caps, EG_params floor, synthesis
filenames, optional rung gates). Harness code dispatches by field name; change
numbers/inventory without re-authoring classifiers.

## What is externalized

- **Screening primary** (default smoke latency, decrease) + non-regression leaves
- **Promotion primary** (held-out quality, increase, locked eval / multi-seed flags)
- **Cadence** `screening_cycles_per_promotion` (default 3 screening : 1 promotion)
- **Exhausted identity fields** (claim class, train/eval version, primary, direction, data digest)
- **Recipe-tweak knobs** + null cap → regime-transition pressure
- **Phase A positive rules** (fixture `n` alone, executable unblock, size-match / EG_params)
- **Synthesis loop** action filenames and fail-closed SFT (still enforced in `hillclimb`)
- **Optional I10 rung gates** (disabled by default)

## Runtime wiring

| Surface | Behavior |
| --- | --- |
| `scripts/run_autotrain_continuous.py` | Cycle role from cadence; Phase A uses `classify_positive_metrics` |
| `scripts/autoresearch.py` hypothesize | Loop-scoped exhausted ledger + recipe-null regime pressure |
| `scripts/autoresearch.py` feedback | Records nulls with policy identity + recipe_null reason when pure recipe |
| `hillclimb.py` | Shared direction-signed effect, synthesis SFT gate, EG_params |

## Champion queue (continuous learning path)

Continuous screening used to thrash the same lever bank every cycle even after
a quality-held Phase A win. The driver now keeps a **loop-local champion queue**
so sticky knobs get a confirmatory retest before more thrash.

| Field | Value |
| --- | --- |
| Ledger | `outputs/autoresearch/loops/<loop_id>/champion_queue.jsonl` |
| Schema | `autotrain_champion_queue/v1` |
| Enqueue | Phase A `positive` **and** quality signal (`quality_held:` / `quality_metric_win:`) with a real lever (`grammar_completion_bounds` or `compact_active_canvas`) |
| Confirm | Next cycle matrix is control + `-confirm` only — **same levers, new seed** (cadence role/suites unchanged) |
| Resolve | Confirm re-holds quality → `confirmed`; else `rejected` (queue head advances) |
| Dedup | Same lever fingerprint already `queued`/`confirming`/`confirmed` is not re-enqueued |

`cycle_intent` on `sdlc_delivery.json` is `confirm` during retests (cadence
`cycle_role` remains `screening`|`promotion`). Pure latency greening without a
quality hold never enters the queue.

## Content digest

`climb_policy_content_digest(payload)` hashes the policy body so version stamps
and tests can detect volatile field changes without re-reading code.

## Related

- [`experiment-campaign-governance.md`](experiment-campaign-governance.md) — claim-class climb gates
- [`autoresearch-autotraining.md`](autoresearch-autotraining.md) — closed loop
