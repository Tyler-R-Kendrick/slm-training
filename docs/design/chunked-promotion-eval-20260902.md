# P11 — promotion tier fits the run cap by chunked, resumable evaluation (2026-09-02)

**Claim class:** harness mechanism at fixture scale. **Not a ship claim.**
No model is promoted, no gate is changed, and every number below comes from a
3-step scratch checkpoint or a mocked clock.

## Problem

`policy.v3.json` (v15) locks the promotion measurement at
`promotion_suite_n: 24`, `promotion_decode_timeout_seconds: 24`,
`promotion_suites: [smoke, held_out]`, primary `held_out.structural_similarity`,
and the power gate is decisive only at n ≥ 6. Decode costs 23–35 s per record
on this host, so 24 records ≈ 576–840 s, while every run is capped at
`MAX_RUN_MINUTES = 3` (`MAX_HARNESS_WALL_SECONDS = 155`). A promotion arm's
`autoresearch run --execute` wall is even smaller (≈ 70 s shared with
training), so every promotion measurement ended `measurement_incomplete` and
the tier could never be decisive. The cap must not change; a timed-out run is
never evidence.

## Mechanism

### 1. Per-record persistence and resume (`eval_runner.py`, `scripts/evaluate_model.py`)

`evaluate(..., partial_scoreboard=True, resume_from=<run_dir>,
max_records_this_run=k)` writes `eval_<suite>.partial.json`
(`EvalPartialScoreboardV1`) atomically after every decoded chunk: prediction,
evidence, decode stats, timeout flag, latency and the per-record quality
leaves. A later run with the same identity — checkpoint SHA-256, eval suite
manifest SHA, `eval_limit`/`eval_offset`, evaluation-policy digest and the
exact record-id list — replays stored records through the normal scorer
without re-decoding; any mismatch restarts from record zero
(`resume_rejected: <key>_mismatch`, fail closed on evidence reuse).

A chunk stops cleanly at `max_records_this_run` newly decoded records or when
the next record cannot fit the evaluation wall at the **full** locked timeout
(`_partial_stop_reason`). The fair-share timeout is computed only over the
records this run will decode (`wall_fit_n`), so the locked per-record timeout
is never shrunk by records left pending; a baked `generate_batch_size` batch is
capped by the remaining budget/wall fit rather than refused.

Undecoded records are `pending`: they count into `incomplete_document_n` (never
a false quality 0), `completed_document_n` excludes them, `measurement_complete`
is `false`, ship gates / AgentV publication are skipped, and the suite stays
non-cacheable evidence until a later run merges it to completion. The merged
final scoreboard has `completed_document_n == n` (minus real decode timeouts,
which persist and replay as timeouts).

CLI: `--partial-scoreboard`, `--resume-run <run_dir>` (defaults to the run's
own directory), `--max-records-this-run k` (env `SLM_EVAL_PARTIAL_SCOREBOARD`,
`SLM_EVAL_RESUME_RUN`, `SLM_EVAL_MAX_RECORDS_THIS_RUN`). When thresholds or
`--ship-gates` were requested and records remain, the process exits
`EXIT_RESUME_PENDING = 10` — no verdict exists yet.

### 2. Locked chunk plan on the promotion manifest (driver, `ExperimentCampaignV1`)

Before any arm executes, the cycle computes `_promotion_chunk_plan(policy,
root)` and stamps it as `measurement_chunk_plan` (`promotion_chunk_plan/v1`,
validated; legacy lock digests tolerate its absence) on every promotion
manifest:

```
per_record_seconds = max(decode_timeout, measured_p95 * (1 + p95_margin))
records_per_run    = floor((MAX_HARNESS_WALL_SECONDS - eval_overhead_seconds) / per_record_seconds)
total_record_n     = Σ_suites min(promotion_suite_n, available records)
run_n              = ceil(total_record_n / records_per_run)      # the locked chunk budget
```

`measured_p95` is the newest `latency_ms_p95` from `eval_held_out.json` /
`eval_smoke.json` under the campaigns root (source path recorded); unmeasured
falls back to the locked timeout. With p95 = 30 s, timeout 24 s, overhead 8 s:
`records_per_run = 4`, and a single 24-record suite needs 6 bounded runs
instead of one 720 s run.

The same plan feeds the promotion arms' typed knobs — `eval_limit =
promotion_suite_n`, `eval_partial_scoreboard = true`,
`eval_max_records_this_run = records_per_run` (measurement keys, excluded from
lever signatures) — which `compile_commands` routes to `evaluate_model`. The
in-arm eval is therefore chunk 1; `execute_commands` types its exit 10 as
`stopped` / `resume_pending` (never `failed`, never a model verdict).

### 3. Sequential bounded chunk runs (driver `_run_promotion_eval_chunks`)

After the arms, for each executed arm in order, the driver rebuilds the arm's
exact `evaluate_model` command from the locked knobs (same checkpoint, suites,
timeout, eval limit), sets `--partial-scoreboard --resume-run <run_dir>
--max-records-this-run records_per_run --evaluation-wall-seconds 155`, and
launches it as its own `MAX_RUN_SECONDS`-bounded subprocess until the merged
`scoreboard.json` reports `measurement_complete: true` or `run_n` runs are
spent. Every launch and its outcome (exit, decoded/pending counts) goes to
`<campaign>/promotion_chunks.json`. The cycle clock is paused for this stage:
each chunk is an independent bounded run under the repository cap, and the
closeout stages keep the budget they had.

### 4. Disposition honesty

* `chunk_budget_exhausted` → `measurement_incomplete:<arm>:chunk_budget_exhausted:runs=k/run_n`
  in the delivery; `_resolve_promotion_result` disposes
  `promotion_inconclusive` (retryable, attempt refunded, stamped
  `last_measurement_incomplete`) — never `promotion_failed`.
* A partial scoreboard is refused by the resolver the same way
  (`measurement_incomplete:<arm>:partial_scoreboard:pending=n`); the metric
  certificate is never disposed on partial evidence.
* `power_feasibility` is re-evaluated at the **final merged n**: the smaller
  `completed_document_n` of the primary suite across control and candidate
  (an upper bound on usable sign-test pairs). The locked pre-run report is
  kept as `locked_n` / `locked_decisive`. Missing or partial scoreboards leave
  the locked report untouched so the incomplete path decides.

## Why this satisfies the run cap

Every process the mechanism starts is a single `scripts.evaluate_model`
invocation bounded by `run_bounded_process` (interrupt at 170 s, eval wall
155 s) or a `train`+chunk-1 arm under the existing arm wall. No run is
extended; a suite that needs 720 s of decode is decoded as ≥ 6 runs of ≤ 155 s
with the partial scoreboard as the only state carried between them. A run that
is killed mid-chunk loses at most the chunk in flight (records already
persisted replay). The total is bounded by the locked `run_n` — exhaustion is
reported, not hidden.

## Evidence (fixture scale)

Mocked-clock tests (`tests/test_harnesses/model_build/test_eval_runner_resume.py`,
24-record suite, 30 s stub decoder, `decode_timeout_seconds = 40`):

| Test | Result |
| --- | --- |
| 24 records at `max_records_this_run = floor(155/30) = 5` | 5 runs, each ≤ 155 s, 24 decodes total, merged `completed_document_n = 24`, one scoreboard; every intermediate suite non-cacheable |
| Chunked vs one-shot | identical `n`, rates, `structural_similarity`, latency percentiles, detail order |
| Run killed at record 4 (KeyboardInterrupt) | 3 records persisted; resume decodes 5 new records (`smoke-03..07`), replays 3, none re-decoded |
| Wall stop (deadline 155 s, timeout 30 s) | 5 records at the full 30 s effective timeout, `stop_reason = evaluation_wall` |
| Checkpoint mismatch | `resume_rejected = checkpoint_sha256_mismatch`, restart from record 0 |
| 3 runs × 5 records | `measurement_complete = false`, `completed 15 / incomplete 9 / pending 9`, non-cacheable |
| Timeouts | persist as incomplete and replay as timeouts |
| `evaluate_suites` two suites, shared budget 6 | gates deferred until complete; second run finishes with gates |
| Baked batch 16, budget 2 | batch capped to `[00, 01]`, then `[02]`; wall fit caps a batch to 5 |

Driver tests (`tests/test_scripts/test_run_autotrain_continuous_chunked_promotion.py`,
stubbed `_stage_command` that advances a fake scoreboard by `records_per_run`):
plan arithmetic (p95 30 s → 4 records/run, 12 runs for 48 records; unmeasured
→ timeout fallback; `min(24, 5)` for a 5-record held_out), manifest stamping
(promotion only; explicit cycle plan verbatim; round-trips through
`ExperimentCampaignV1`), promotion knobs lock suite n and per-run cap, 24
records complete in 5 sequential bounded launches per arm with one merged
scoreboard, exhausted budget (3 × 5) → `measurement_incomplete` +
`promotion_inconclusive` inputs, killed-mid-chunk resume skips stored records,
missing checkpoint typed, merged-n power feasibility (24 → decisive; 5 →
`promotion_infeasible_by_design:n=5`).

Real CLI path (`scripts.evaluate_model`, CPU, this host):

| Run | Command shape | Wall | Result |
| --- | --- | --- | --- |
| scratch train | `train_model --model twotower --steps 3 --d-model 32 --context-layers 1 --denoiser-layers 1 --context-backend scratch` on `wf_smoke_v2` | 10.1 s | current-contract checkpoint (scratch run dir outside the repo) |
| chunk 1 | `--suite smoke --eval-limit 3 --partial-scoreboard --max-records-this-run 2 --decode-timeout-seconds 24` | 13.6 s | `decoded_this_run_n 2`, `pending 1`, `completed_document_n 2 / incomplete 1`, `measurement_complete false`, `stop_reason record_budget`, exit 0 |
| chunk 2 | same + `--resume-run <run_dir> --fail-under-parse-rate 0.5` | 10.2 s | `replayed 2`, `decoded 1`, `pending 0`, `completed_document_n 3 == n`, `measurement_complete true`, `resumed_from_partial` on 2 detail rows, exit 0 |

Scratch metrics (`parse_rate 1.0`, `structural_similarity ≈ 0.10`,
`latency_ms_p95 ≈ 6.5 s`) are a 3-step model's noise — they exist only to show
the resume path decodes, persists and merges real records.

**Deviation from the card:** the committed demo checkpoint
`src/slm_training/resources/checkpoints/playground_demo/last.pt` is output
contract v0 and refuses to load under the required `symbol_only/v2`
(`OutputContractError`, also recorded in `docs/MODEL_CARD.md`). The CLI path was
exercised on a freshly trained current-contract scratch checkpoint instead; the
contract was not weakened.

## Honesty notes

* Fixture scale throughout. Nothing here changes `PARSE_RATE_PERFECT`, any
  ship gate, or a locked endpoint/arm/seed.
* The local default eval version (`e938_role_safe_all_targets_smoke24_v1`) has
  **5 held_out records**. The plan reports `min(24, 5)`; the merged-n power
  report will therefore be **not decisive (n = 5 < 6)** at resolve time even
  when every chunk completes, and the disposition is
  `promotion_infeasible_by_design` — the correct verdict for that suite, not a
  harness failure. A decisive promotion tier needs a held_out suite with ≥ 6
  records; growing it is data work, not a gate change.
* `merged_n` is a bound (smaller completed arm), not a paired-id intersection.
* The chunk stage pauses the cycle clock; total promotion wall is bounded by
  `run_n × MAX_RUN_SECONDS` per arm and reported in `promotion_chunks.json`.
* Decode p95 is taken from the newest eval on disk; a stale or faster host
  changes `records_per_run` for the *next* plan only — a plan is never edited
  after execution starts.

## Version stamps

`harness.model_build.eval` v102, `harness.autoresearch.experiment_campaign`
v272 (`src/slm_training/resources/versions.json`).
