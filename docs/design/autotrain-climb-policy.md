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
proof-driven **and effect-gated** (cert continue is necessary but not sufficient):

| Gate | Contract |
| --- | --- |
| Locked expectations | `metric_expectations.promote.v1.json` SHA-256 bound on the promote campaign as `metric_expectations_sha256` **before** outcomes; dispose **fails closed** if digest missing/unreadable or mismatches the certificate |
| Formal preflight | Required template `metrics.structural_similarity_monotone`; content-addressed artifact `artifacts/formal_preflights/<sha>.json` bound into obligations. Promote experiment carries `formal_claims` **inside the hypothesis matrix** (before lock) so execute membership stays exact — do **not** rewrite experiments post-hypothesize. Train only when formal status is `proved`. Continuous Lean wall is **600s**. **Timeout → `timed_out` / `promotion_inconclusive`**. Pre-warm OpenUIProofs when possible. |
| **Primary effect** | Dual-arm policy `promotion_primary` (default `held_out.structural_similarity`) must improve by more than `minimum_effect` (default **0.01**). Parse non-regression when both arms measure parse_rate. Null / insufficient delta → `promotion_failed` (model/effect reject), not harness. Policy knobs: `promotion_dispose.*`. |
| Certificate | Continuous **exports** LeverProof `metric-certificate.json` from control **and** candidate suite metrics; disposition via `optimum_feedback`. **`continue` alone never authorizes `promoted`.** |
| Phase A metrics | Promotion role loads **`eval_held_out.json`** for the policy primary |
| Thrash skip | Only arms **currently open** in the funnel are deprioritized |

### Promote dispositions (do not conflate)

| Status | Meaning | Retry? |
| --- | --- | --- |
| `promoted` | Formal proved + **held-out primary effect** + cert v2 + `optimum_feedback=continue` | No (done) |
| `promotion_failed` | Complete measurement; null primary, cert/policy miss, or formal unproved (non-timeout) model/proof/effect reject | Limited |
| `promotion_inconclusive` | Formal **timeout** — incomplete measurement | Yes |
| **`harness_failure`** | Matrix membership, execute abort, missing promote run, cert incomplete **because candidate never ran** — **not a model result** | Yes |
| `rejected` | Confirm retest quality fail | No (confirm path) |

Learning events append to
`loops/<loop_id>/learning_certificate_ledger.jsonl`. Screening thrash may still
generate observations; it cannot authorize promotion. Ship gates and
`fixture_insufficient_n_alone_not_positive` are unchanged.

### Driver singleton (ops)

Exactly **one** `run_autotrain_continuous` process may own a `loop_id`. Lock:
`outputs/autoresearch/loops/<loop_id>/driver.lock` (fcntl exclusive). A second
start prints `DRIVER_ALREADY_RUNNING` and exits 2. Reclaim is automatic when
the owner process exits (kernel drops the flock). Do not dual-start the same
loop; duplicate drivers race cycle indices and empty campaigns.

### Model-build ladder (honest)

| Level | What continuous does | What it does not |
| --- | --- | --- |
| L1 train checkpoint | `train_model` → `last.pt` under campaign runs | HF bucket sync (`sync_checkpoints:false`) |
| L2 learn / promote lever | Champion queue + proof disposition | Ship gates / multi-suite n≥20 |
| L3 ship model | Finite `slm sft train` / `hf_jobs_train` + MODEL_CARD | Not the continuous thrash loop |

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
