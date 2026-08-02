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

- **Screening primary** (`smoke.structural_similarity`, increase) + parse and binder-reference non-regression
- **Promotion primary** (held-out quality, increase, locked eval / multi-seed flags)
- **Cadence** `screening_cycles_per_promotion` (default 3 screening : 1 promotion)
- **Exhausted identity fields** (claim class, train/eval version, primary, direction, data digest)
- **Recipe-tweak knobs** + null cap → regime-transition pressure
- **Phase A positive rules** (fixture `n` alone, executable unblock, size-match / EG_params, minimum efficiency effect)
- **Synthesis loop** action filenames and fail-closed SFT (still enforced in `hillclimb`)
- **I10 rung gates** (enabled and fail-closed, with durable prior-rung evidence)
- **Command walls** (screening, promotion, and Lean obey `MAX_RUN_MINUTES`)
- **Frozen replay limit** (identical incomplete replays before harness repair)

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
resume state use `AutotrainLoopStateV1` at `loops/<loop_id>/state.json`. While an
arm runs, start/heartbeat callbacks publish its process-group leader as
`child_pid` plus `stage_started_at`, so the report reflects live process truth. The
same state-aware bounded wrapper owns Git synchronization, campaign init,
research, hypothesis compilation, status, Lean preflight, and certificate checking;
there are no unsupervised child-process islands in the driver. The
handoff separates climb from ship state and carries evidence-bound actions that
name their owner skill. Three consecutive hard failures with the same fingerprint
mark the loop blocked instead of printing and sleeping forever; timeouts, transient
fetch failures, and incomplete measurements remain retriable and never accumulate a
hard-block count.
It also enumerates checkpoints created during the cycle and requires the
supervisor to update the model card and README summary before continuing.
Prerequisite actions are closed by append-only, action-digest-bound receipts whose
typed evidence SHA is revalidated on every read. Historical URI-only receipts do
not authorize a successor. A theorem-backed `stop_campaign` and its distinct
`repair_formal` action are both prerequisites. The next supervised campaign fails
closed if required stop, harness, Lean, data, documentation, or merged-delivery
evidence is absent. Partial eval scoreboards are infrastructure evidence and never
enter model-quality comparison.

Structural supervision arms also carry a `structural_aux_head_profile`. The
recommended arm and its zero-loss control prebuild exactly the same auxiliary
heads, so the treatment changes supervision rather than trainable capacity.
Training parameter counts are projected into the typed outcome and terminal
matrix; Phase A falls back to the bound train summary and applies the `EG_params`
gate if any growth remains. A missing parameter count can no longer make a growth
arm look size matched. After the handoff is written, the terminal matrix renders
its post-outcome priorities. A completed null or negative therefore rotates to a
distinct quality hypothesis instead of printing the already-executed preregistration.

Policy v4 adds `positive_classification.minimum_efficiency_gain_fraction=0.05`.
An efficiency-only screen must improve meaningful-programs per millisecond by at
least 5%; smaller positive deltas are typed as
`efficiency_win_rejected_min_effect` and cannot override the structural primary's
minimum effect. c1730's 15.25% screen triggered a new-seed replication, while
c1731 collapsed to 0.66% with identical quality and is rejected. This prevents
CPU jitter from minting positive delivery layers or repeatedly steering the same
exhausted model arm.

`measurement.max_consecutive_frozen_replays` bounds identical incomplete replay
cycles. Exhaustion is not a model rejection: the handoff preserves the frozen
manifest and routes a typed `repair_harness` action to
`improve-openui-harnesses`. The next automatic cycle remains receipt-blocked until
that canonical-owner repair is acknowledged. The exact retry stays queued behind the
repair, whose receipt resets the consecutive count. This prevents an infinite
unmodified timeout replay loop while retaining the exact reproduction input.
Policy v5 sets this allowance to one: the original incomplete measurement plus
one hash-linked, checkpoint-cached reproduction is enough to establish a
repeatable harness failure. A second unmodified replay adds no attribution while
spending the same evaluation wall, so repair becomes mandatory before another
retry.
When AgentV itself finalizes every record disposition and reports one or more typed
internal decode timeouts, the loop does not spend that replay allowance pretending
the supervisor stage is still incomplete. It routes immediately to canonical
`model_build` runtime repair and queues the frozen arm behind that repair. Quality
metrics remain incomplete and non-promotable; the finalized timeout is infrastructure
evidence that identifies the next owner.

The autoresearch executor also passes its remaining stage wall into the canonical
model-build evaluator. The evaluator reserves finalization time and fairly
partitions the remaining wall across unprocessed records, capped by the locked
per-record timeout. A pathological record therefore becomes a typed internal
timeout while later records still receive a bounded attempt; the evaluator can
write its scoreboard and AgentV bundle before the outer supervisor interrupt.
The effective minimum/maximum record allocations are persisted beside the suite
metrics. This is scheduling only: it never widens a timeout, changes a gate, or
turns an incomplete record into quality evidence.

One cycle still obeys the repository hard cap. The bounded wall is partitioned into
three equal planning shares after retaining the canonical finalization reserve: a
promotion-only formal share and equal control/candidate shares. Before Lean or
either arm starts, the driver proves both complete arm shares plus finalization fit;
Lean receives only the remaining time after those reservations. It fails closed
instead of allowing proof startup or the first arm to starve the second.
This avoids stranding orchestration time that a complete multi-record evaluation
needs while keeping decision-bearing evaluation serialized so CPU contention cannot
invalidate latency attribution. Serialized arms are counterbalanced AB/BA by cycle
parity; promotion replicates alternate using the count of already verified
replicates. Every campaign declares the one seed it actually executes. Promotion
evidence is append-only and content-bound to both manifests, the durable delivery,
the metric certificate, and the exact metrics; acceptance requires distinct seeds
and both orderings. Scratch one-shot arms keep the serving checkpoint
but skip the unused full optimizer/RNG state. A long-lived unsupervised driver that
integrates a new HEAD re-executes itself before creating a campaign, preventing stale
imported policy constants from poisoning later cycles.

## Champion queue (continuous learning path)

Continuous screening used to thrash the same lever bank every cycle even after
a quality-held Phase A win. The driver now keeps a **loop-local champion queue**
so sticky knobs get a confirmatory retest before more thrash.

| Field | Value |
| --- | --- |
| Ledger | `outputs/autoresearch/loops/<loop_id>/champion_queue.jsonl` |
| Schema | `autotrain_champion_queue/v1` |
| Enqueue | Phase A `positive` **and** quality signal (`quality_held:` / `quality_metric_win:`) on any registered screening lever — not pure matched control |
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
| Formal preflight | Required template `metrics.structural_similarity_monotone`; content-addressed artifact `artifacts/formal_preflights/<sha>.json` bound into obligations. Promote experiment carries `formal_claims` inside the hypothesis matrix before lock. Train only when formal status is `proved`. This template routes to the Mathlib-free LeverProof theorem and mandatory `make test` audit. A cached proved sidecar has no authority until the artifact is revalidated against campaign/experiment/claim identity, current template, source digests, and proof bundle, and the sidecar records that validated SHA. The Lean wall is the time left after both arms and finalization are reserved; timeout remains inconclusive. |
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
bank and **rotate** `recommended_experiment_id` by cycle index. Alongside the
historical runtime/recipe controls, the bank includes size-matched structural
supervision arms for component plan, component edges, component inventory,
binder topology, and joint component structure. These change loss attribution,
not model capacity. Arms recently in the champion queue are deprioritized, and
saturated causal families are skipped until the integrated code changes.

Policy v3 replaces the saturated `smoke.binder_reference_f1` screening primary
with `smoke.structural_similarity` and requires both parse rate and binder F1 not
to regress. c1728 measured binder F1 at 1.0 in both arms while structure differed
by 0.2099, proving the prior primary could not rank the observed quality failure.
Structural similarity already has the Mathlib-free LeverProof monotonicity theorem
used by promotion; screening remains fixture evidence and cannot bypass the full
promotion proof, multi-seed, or ship gates. A completed frozen replay also rewrites
its stale infrastructure priority to the next distinct model-quality arm.
Policy v4 retains those quality rules and adds the minimum efficiency effect above.

## Content digest

`climb_policy_content_digest(payload)` hashes the policy body so version stamps
and tests can detect volatile field changes without re-reading code.

## Related

- [`experiment-campaign-governance.md`](experiment-campaign-governance.md) — claim-class climb gates
- [`autoresearch-autotraining.md`](autoresearch-autotraining.md) — closed loop
