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
filenames, and rung gates). Harness code dispatches by field name; change
numbers/inventory without re-authoring classifiers.

## What is externalized

- **Screening primary** (binding/reference quality fallback, increase) + non-regression leaves
- **Promotion primary** (held-out quality, increase, locked eval / multi-seed flags)
- **Cadence** `screening_cycles_per_promotion` (default 3 screening : 1 promotion)
- **Exhausted identity fields** (claim class, train/eval version, primary, direction, data digest)
- **Recipe-tweak knobs** + null cap → regime-transition pressure
- **Phase A positive rules** (fixture `n` alone, executable unblock, size-match / EG_params)
- **Synthesis loop** action filenames and fail-closed SFT (still enforced in `hillclimb`)
- **I10 rung gates** (enabled and fail-closed, with durable prior-rung evidence)
- **Command walls** (screening, promotion, and Lean obey `MAX_RUN_MINUTES`)

## Runtime wiring

| Surface | Behavior |
| --- | --- |
| `scripts/run_autotrain_continuous.py` | Cycle role from cadence; Phase A uses `classify_positive_metrics` |
| `scripts/autoresearch.py` hypothesize | Loop-scoped exhausted ledger + recipe-null regime pressure |
| `scripts/autoresearch.py` feedback | Records nulls with policy identity + recipe_null reason when pure recipe |
| `hillclimb.py` | Shared direction-signed effect, synthesis SFT gate, EG_params |

## Agent-supervised control plane

Bare `/autotrain` is owned by an unbudgeted host goal. The agent syncs Git and
runs one bounded driver cycle with `--supervised --max-cycles 1`, then regains
control for repairs, documentation, and delivery. Each cycle writes strict
`AutotrainCycleHandoffV1` to `<campaign>/cycle_handoff.json`; heartbeat and
resume state use `AutotrainLoopStateV1` at `loops/<loop_id>/state.json`. The
handoff separates climb from ship state and carries evidence-bound actions that
name their owner skill. Three consecutive failures with the same fingerprint
mark the loop blocked instead of printing and sleeping forever.
It also enumerates checkpoints created during the cycle and requires the
supervisor to update the model card and README summary before continuing.

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
| Dedup | Same open lever fingerprint is not re-enqueued; fingerprint excludes cycle-local `steps` jitter. A causal family saturates after two terminal attempts on the same integrated code and reopens only after code identity changes. |

### Proof driver (promote authorization)

**Phase A smoke quality-held alone never marks `climb_accepted`.** Climb
acceptance is proof-driven and effect-gated; `ship_promoted` is a separate full
AgentEvals verdict:

| Gate | Contract |
| --- | --- |
| Locked expectations | `metric_expectations.promote.v1.json` SHA-256 bound on the promote campaign as `metric_expectations_sha256` **before** outcomes; dispose **fails closed** if digest missing/unreadable or mismatches the certificate |
| Formal preflight | Required template `metrics.structural_similarity_monotone`; content-addressed artifact `artifacts/formal_preflights/<sha>.json` bound into obligations. Promote experiment carries `formal_claims` inside the hypothesis matrix before lock. Train only when formal status is `proved`. The Lean wall is `MAX_RUN_SECONDS`; timeout remains inconclusive. |
| **Primary effect** | Dual-arm policy `promotion_primary` (default `held_out.structural_similarity`) must improve by more than `minimum_effect` (default **0.01**). Parse non-regression when both arms measure parse_rate. Null / insufficient delta → `promotion_failed` (model/effect reject), not harness. Policy knobs: `promotion_dispose.*`. |
| Certificate | Continuous exports LeverProof `metric-certificate.json` from control and candidate suite metrics; disposition uses `optimum_feedback`. `continue` alone never authorizes `climb_accepted` or ship. |
| Phase A metrics | Promotion role loads **`eval_held_out.json`** for the policy primary |
| Thrash skip | Only arms **currently open** in the funnel are deprioritized |

### Promote dispositions (do not conflate)

| Status | Meaning | Retry? |
| --- | --- | --- |
| `climb_accepted` | Fixture climb: formal proved + held-out primary effect + cert v2 + `optimum_feedback=continue` | No (done) |
| `ship_promoted` | Full authoritative AgentEvals gates, suite/sample floors, rung evidence, parameter efficiency, and publication evidence | No (ship authority) |
| `promotion_failed` | Complete measurement; null primary, cert/policy miss, or formal unproved (non-timeout) model/proof/effect reject | Limited |
| `promotion_inconclusive` | Formal **timeout** — incomplete measurement | Yes |
| **`harness_failure`** | Matrix membership, execute abort, missing promote run, cert incomplete **because candidate never ran** — **not a model result** | Yes |
| `rejected` | Confirm retest quality fail | No (confirm path) |

Learning events append to
`loops/<loop_id>/learning_certificate_ledger.jsonl`. Screening thrash may still
generate observations; it cannot authorize ship promotion. Ship gates and
`fixture_insufficient_n_alone_not_positive` are unchanged.

### Driver singleton (ops)

Exactly one driver process may own a `loop_id` while a bounded cycle runs. Lock:
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
`promotion` (cadence `cycle_role` remains `screening`|`promotion`). Latency is
a cost/non-regression signal; the screening primary is quality.

## Thrash rotation (matrix diversity)

Screening (and promotion-without-confirmed) matrices include the full lever
bank (`bounds`, `canvas`, `both`, `steps`, `batch1`) and **rotate**
`recommended_experiment_id` by cycle index. Arms recently in the champion
queue are deprioritized, and saturated causal families are skipped until the
integrated code changes.

## Content digest

`climb_policy_content_digest(payload)` hashes the policy body so version stamps
and tests can detect volatile field changes without re-reading code.

## Related

- [`experiment-campaign-governance.md`](experiment-campaign-governance.md) — claim-class climb gates
- [`autoresearch-autotraining.md`](autoresearch-autotraining.md) — closed loop
