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
| Enqueue | Phase A `positive` **and** quality signal (`quality_held:` / `quality_metric_win:`) on any thrash lever (`bounds` / `canvas` / `both` / `steps` / `batch1`) — not pure matched control |
| Confirm | Next cycle matrix is control + `-confirm` — **same levers, new seed** (cadence role/suites unchanged); max **2** confirm attempts then `rejected` |
| Promote | On **promotion cadence**, a `confirmed` head becomes control + `-promote` under promotion suites/seeds |
| Dedup | Same lever fingerprint already `queued`/`confirming`/`confirmed` is not re-enqueued; fingerprint **excludes** cycle-local `steps` jitter |

### Proof driver (promote authorization)

**Phase A smoke quality-held alone never marks `promoted`.** Promotion is
proof-driven:

| Gate | Contract |
| --- | --- |
| Locked expectations | `metric_expectations.promote.v1.json` SHA-256 bound on the promote campaign as `metric_expectations_sha256` **before** outcomes; dispose **fails closed** if digest missing/unreadable or mismatches the certificate |
| Formal preflight | Required template `metrics.structural_similarity_monotone`; content-addressed artifact `artifacts/formal_preflights/<sha>.json` bound into obligations; promote experiment carries matching `formal_claims`; train only when status is `proved` (else skip execute + `promotion_failed`) |
| Certificate | Continuous **exports** LeverProof `metric-certificate.json` from control/candidate suite metrics (per-mille SS + parse) via in-repo `leverproof-lean check`; disposition via `optimum_feedback` |
| Thrash skip | Only arms **currently open** in the champion funnel are deprioritized — rejected/promotion_failed do **not** permanently starve bounds/canvas |
| `continue` (all in band) | only path to **`promoted`** |
| `stop` (theorem miss) | `promotion_failed`; no five-lane thrash |
| `block_promotion_and_diagnose` (assumption miss) | `promotion_failed` + `five_lane_successor_matrix.json` (measurement_control, training_method, architecture, lean_model, assumptions) |
| Missing / v1 / digest mismatch | `promotion_failed` (`promote_requires_certificate…`) |

Learning events append to
`loops/<loop_id>/learning_certificate_ledger.jsonl`. Screening thrash may still
generate observations; it cannot authorize promotion. Ship gates and
`fixture_insufficient_n_alone_not_positive` are unchanged.

`cycle_intent` on `sdlc_delivery.json` is `confirm` / `promote` / `screening` /
`promotion` (cadence `cycle_role` remains `screening`|`promotion`). Pure latency
greening without a quality hold never enters the queue.

## Thrash rotation (matrix diversity)

Screening (and promotion-without-confirmed) matrices include the full lever
bank (`bounds`, `canvas`, `both`, `steps`, `batch1`) and **rotate**
`recommended_experiment_id` by cycle index. Arms recently in the champion
queue (queued/confirmed/rejected/…) are deprioritized so the loop does not
forever recommend the same lever after a reject.

## Content digest

`climb_policy_content_digest(payload)` hashes the policy body so version stamps
and tests can detect volatile field changes without re-reading code.

## Related

- [`experiment-campaign-governance.md`](experiment-campaign-governance.md) — claim-class climb gates
- [`autoresearch-autotraining.md`](autoresearch-autotraining.md) — closed loop
