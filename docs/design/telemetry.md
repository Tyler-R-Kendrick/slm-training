# Cycle telemetry (train / inference bottlenecks)

## What it measures

`slm_training.runtime.telemetry.CycleTelemetry` accumulates named spans and ranks them by
wall-time share. Artifacts:

| Artifact | Source |
| --- | --- |
| `outputs/runs/<id>/train_telemetry.json` | `train_loop.train` (default on; `--no-telemetry` to disable) |
| `outputs/runs/<id>/rl/rl_telemetry.json` | `scripts/train_rl.py` / GRPO |
| `docs/design/cycle-telemetry.json` | `scripts/bench_telemetry.py` scratch profile |
| `outputs/runs/<id>/run_insights.json` | deterministic loss/collapse analysis, phase guidance, and optional generated hypotheses |

Every new run also has `outputs/runs/<id>/trace.json`, which points to one
central W3C-correlated bundle:

```text
outputs/traces/<trace-id>/
  manifest.json
  signals/traces/<service-instance>.otlp.jsonl
  signals/logs/<service-instance>.otlp.jsonl
  domain/<kind>/*.jsonl
```

The signal shards use OTLP JSON encoding and standard service resource
attributes. Logs carry the current trace and span IDs. Detailed decode canvases,
Molt rollouts, and synthesis rows remain linked domain JSONL instead of being
placed in network telemetry bodies. Run-local telemetry files are derived
summaries; the trace bundle owns the raw correlated signals.

Set `OTEL_EXPORTER_OTLP_ENDPOINT` (or a signal-specific endpoint) to mirror the
same JSON payloads through OTLP/HTTP. Local persistence remains authoritative if
the remote endpoint is missing or unavailable.

## Shared telemetry peers (active runs + live streaming)

Every web app instance embeds an in-memory OTLP hub (`slm_training/web/otel_hub.py`)
speaking two protocol surfaces, which together make any instance a **telemetry
peer**:

- **Ingest** — standard OTLP/HTTP JSON at `POST /v1/traces` and `POST /v1/logs`
  (exactly the paths `RunTrace._mirror` derives from a base endpoint URL).
- **Read** — `GET /api/otel/runs` (active-run list), `GET
  /api/otel/runs/<id>/events?since=<seq>` (cursor page), and `GET
  /api/otel/runs/<id>/stream` (SSE: `status` / `otel` / `dropped` / `ping` /
  `error` frames, each stamped with the hub's boot `hub_epoch`).

There is no blessed central server: "the shared endpoint" is whichever
always-reachable peer a team agrees on. `SLM_OTEL_PEERS` (comma-separated URLs)
wires a machine into the mesh:

- **Broadcast** — trainers mirror through `OTEL_EXPORTER_OTLP_ENDPOINT` if set,
  else the **first** peer in `SLM_OTEL_PEERS` (single blocking 2s-timeout POST
  per record keeps the training hot path bounded). The train loop additionally
  emits a throttled `train.progress` log record (step, loss, target tokens)
  every `SLM_OTEL_PROGRESS_SECONDS` (default 20; `0` disables) so streams show
  live training activity between run start and end.
- **Read federation** — the dashboard's active-runs list merges local ingest ∪
  every peer (in listed order) ∪ a zero-config disk fallback
  (`outputs/runs/*/metrics.jsonl` touched in the last 10 minutes, list-only),
  deduped by run id with that precedence. Federation always requests a peer's
  **local-only** view (`?local=1`), so cyclic peer graphs (A↔B) are loop-safe by
  construction — reads fan out at request time, nothing is re-broadcast.
- **Laziness** — peer fetches happen only while a client request or stream is
  attached; per-run SSE fan-out queues exist only while someone subscribes; the
  dashboard opens a run's EventSource only while the observing page is mounted
  and the tab visible (hidden tabs close it and resume from the last `seq`).

Run lifecycle at a peer: `run.started` → **active**; `run.completed` /
root-span status 1 → **completed**; `run.failed` / status 2 → **failed**; no
events for 10 minutes → **stale** (any later event revives, which also covers
hub restarts — state is in-memory only and `hub_epoch` tells clients to reset;
durable history stays in the producer's local trace bundle).

**Auth (ingest only).** `SLM_OTEL_AUTH` selects the mode: `open` (default when
nothing is configured — keeps localhost zero-config), `token` (bearer must match
`SLM_OTEL_TOKEN`; senders inherit it automatically), or `hf` (bearer validated
against `https://huggingface.co/api/whoami-v2`, and the resolved username is
stamped on the run so the dashboard shows *who* is running what). With
`SLM_OTEL_AUTH=hf` a sender forwards its `HF_TOKEN` — explicit opt-in only, and
prefer a fine-grained token. `OTEL_EXPORTER_OTLP_HEADERS` (`k=v,k2=v2`)
overrides sender headers outright. Reads are tokenless like every other
observability endpoint (`EventSource` cannot send headers) — anyone who can
reach a peer can read run telemetry metadata, so front a private deployment with
network controls if that matters.

**Deployment constraints.** The hub is in-memory and single-process: run peers
with a single uvicorn worker (the default `scripts/serve_playground` path). On
serverless deploys (Vercel) the hub disables itself — ingest and local streams
return 503, `capabilities.otel.hub` is `false`, but the merged list still
read-through-federates configured peers. A rendezvous peer for a team is just a
persistently hosted instance: a lab box, a VM, or e.g. a Docker-SDK Hugging Face
Space running `uvicorn` on one worker with `SLM_OTEL_AUTH=hf` set — no dedicated
artifact required.

## Spans

**Train:** `batch_build`, `forward` → nested `context_encode` + `denoiser_forward`,
`backward`, `optim_step`, `eval_suites`, `device_sync`, `final_save`.

**Generate:** `generate_batch` → `generate_once` / `best_of_n_rank`, plus
`context_encode` inside the model.

## Decode-stats deterministic-row metrics

These counters share the existing `DecodeStats` envelope and aggregate under
`metrics["decode_stats"]`:

| Field | Unit and meaning |
| --- | --- |
| `denoiser_rows_evaluated` | Rows actually evaluated by denoiser/backbone calls. |
| `ambiguous_rows_forwarded` | Active rows whose current decision required model ranking. |
| `forced_row_tokens_without_forward` | Exact row-token decisions committed without neural evaluation. |
| `all_forced_steps_without_forward` | Decode steps where every live row was exact and no neural call ran. |

`forced_tokens` / `forced_spans` retain their compiler/choice meanings. P3 tokens
accepted from already-computed logits remain `accepted_run_tokens`; they are not
reported as no-forward proof decisions. Binding evidence contains keys, slot ids,
digests, and byte counts but never raw caller content.

## Interrupt-safe decode progress

When the bounded-process supervisor interrupts an evaluation, the model-build
evaluator atomically writes `<run-root>/<id>/decode_progress.json`.
`DecodeProgressV1` is a bounded, version-stamped diagnostic sidecar with
processed-record count, active record ids, and the aggregate `DecodeStats` observed
so far. It is explicitly
`measurement_complete=false` and `scoreable=false`: neither autotrain nor promotion
may treat its counters as quality or performance results. A successful canonical
`eval_<suite>.json` write removes the transient sidecar.

If the bounded-process supervisor interrupts the evaluator, the active stats bucket
is attached to the exception, completion-session deltas are folded in `finally`, and
the sidecar is refreshed before the interrupt propagates. Autoresearch copies only a
fresh sidecar into `stage_telemetry[].partial_output`; stale files are rejected by
revision, and malformed, scoreable, complete, or wrong-run payloads are ignored.
This preserves grammar-work and prefill counters for diagnosis while the experiment
outcome remains stopped and metric-empty.

## Decode-stats solver work metrics (VSS1-04 / SLM-64)

The verified solver's per-decode work is measured on the existing
[`DecodeStats`](../../src/slm_training/models/decode_stats.py) envelope (not a new
owner). All fields default to zero on every historical/default path (solver
disabled), and solver wall time is separated from `denoiser_ms` / `projection_ms`.
Stable names:

| Field | Meaning |
| --- | --- |
| `solver_ms` | Solver wall time (`timed_ms`), separate from denoiser/projection. |
| `solver_enabled` | `1` when the solver ran on a decision, else `0`. |
| `solver_closure_passes` | Exact-closure fixed-point passes. |
| `solver_support_queries` / `solver_support_cache_hits` | Support-oracle queries and request-local cache hits. |
| `solver_supported` / `solver_unsupported` / `solver_unknown` | Tri-state support verdict counts. |
| `solver_certified_removed` | Candidates removed by replay-valid certificates. |
| `solver_decisions` / `solver_backtracks` / `solver_nogoods` | Reversible-search work (controller path). |
| `solver_expanded_nodes` / `solver_verifier_calls` | Enumeration nodes and verifier calls. |
| `solver_certificate_replay_failures` | Certificate replays that failed (0 at decode — closure never removes on a failed replay; populated by offline trace audits). |
| `solver_terminal_status` | Honest terminal: `unknown` / `certified_unsat` / `budget_exhausted` (closure never claims `solved`). |

They surface **only** under `metrics["decode_stats"]` in `eval_<suite>.json` (and,
transitively, `scoreboard.json`) via `aggregate_stats`; no new top-level metric
keys or files. They do not overload the existing grammar/lattice candidate
counters.

## Typed decode-stats records (PCT-001 / SLM-439)

[`DecodeStatsRecordV1`](../../src/slm_training/models/decode_stats.py) is an
optional, tamper-evident envelope built from an existing `DecodeStats`
snapshot -- an extension of `decode_telemetry`, not a second telemetry owner.
It never mutates `DecodeStats` or `aggregate_stats`, so every current reader
(`durable_decode_stats` in `run_quality_matrix.py`, the `decode_headlines`
allowlist in `autoresearch/engine.py`, the `decode_progress.json` sidecar
gate) is unaffected.

A record adds:

* **`measurement_stage`** -- one of `process_cold` / `artifact_cold` /
  `model_cold` / `request_cold` / `steady_state`. A "steady state" aggregate
  must filter on this field (`completed_steady_state`) so a cold-start
  decode's inflated `total_ms` can never silently blend into it.
* **`completeness`** -- `complete` / `partial_timeout` / `aborted`. Only
  `complete` records may feed a "measured performance" summary.
* **`legal_domain_size` / `legal_domain_status`** -- the latter mirrors
  `CompletionDomainV1.status` (`complete` / `incomplete` / `unsupported` /
  `unknown`) rather than inventing a second symbol-table completeness
  vocabulary.
* **`witness_ids`** -- opaque references (e.g. a `VerifierWitnessV1.witness_digest`
  from `evals/semantic_failure.py`) linking a decode to independently
  replayable verifier evidence, when one was produced.
* **`identity`** -- a `DecodeIdentityV1` binding to contract version, pack,
  tokenizer, artifact/checkpoint sha256, evaluator version/hash, and code
  commit; every field is optional and explicit, never inferred.
* **`record_digest`** -- a canonical-JSON sha256 over every other field,
  following the same digest convention as `dsl/solver/replay.py` and
  `evals/semantic_failure.py`'s `VerifierWitnessV1`. `from_dict` recomputes
  and compares it, raising (fail closed) on drift or corruption.

`proves_zero_neural_work(stats)` / `proves_zero_search_work(stats)` give the
I2 `forwards_count == 0` bypass proof (and its search-side analog) a reusable,
directly testable form.

`append_decode_stats_record` / `iter_decode_stats_records` give records an
append-only, fsync'd JSONL home with a `prev_record_digest` hash chain: a
reordered, deleted, or tampered line breaks the chain and `iter_decode_stats_records`
raises rather than silently replaying a corrupted log.

**Scope note**: this PR ships the typed contract, builder, and persistence
layer with full unit coverage. It does not yet wire `build_decode_stats_record`
into any live `collect_decode_stats` call site (`eval_runner.py`,
`decode_progress.json`, `scoreboard.json`) -- that integration, plus the
`durable_decode_stats` allowlist and `decode_headlines` updates it would need,
is deliberately left to PCT-003 (honest end-to-end cold/warm benchmark
harness) and PCT-007 (mechanism activation/disposition telemetry), which own
that wiring.

## How to use

```bash
# Microbench scratch train+generate
python -m scripts.bench_telemetry --train-steps 12 --gen-prompts 8

# Production train writes telemetry into the run dir
python -m scripts.train_model --fast-train --steps 200 --run-id tel_demo

# Inspect bottlenecks
python -c "import json; print(json.load(open('outputs/runs/tel_demo/train_telemetry.json'))['bottlenecks'])"
```

Interpret `bottlenecks[0]` as the primary hot spot. On HF trains, expect
`context_encode` to dominate until the frozen-backbone cache warms; on scratch,
`denoiser_forward` / `backward` usually lead.

## Run insight report

Completed training and performance-matrix runs write `run_insights.json` beside
their other artifacts. The report contains a bounded loss series for charting,
deterministic collapse indicators, and phase-specific optimization suggestions.
The current indicators flag non-finite values, robust rolling-baseline spikes,
suspicious abrupt drops, and sustained divergence. They are diagnostic signals,
not proof of a cause; each marker retains the observed step and a bounded follow-up
experiment.

The Smoke page links each matrix row to the compiled run-detail page. That page
shows loss over time, marks detected collapse episodes, and exposes phase guidance
through accessible tooltips. On first view of a completed run, browser inference is
enabled by default and may add a hypothesis layer using the browser LanguageModel
API or the existing Transformers.js fallback. Users can disable it. Server-side
OpenAI enrichment is a separate opt-in fallback and is available only when
`OPENAI_API_KEY` is configured; Responses are structured and use `store=False`.
The deterministic report remains useful when no model is available.

## PostHog mirror (WP-6: LLM analytics + dashboard error tracking)

The self-hosted OpenTelemetry stack stays the source of truth: spans, logs,
and OTLP JSONL bundles are written locally by
`src/slm_training/runtime/telemetry/trace.py` and aggregated by the in-memory
OTLP hub (`src/slm_training/web/otel_hub.py`). PostHog is a **mirror only** —
a fire-and-forget copy of a small set of LLM analytics events so runs show up
in PostHog's free-tier LLM analytics and error-tracking views. Nothing reads
PostHog back, no scheduler or CI workflow is involved, and losing every
mirrored event loses no evidence.

### Architecture

- `src/slm_training/runtime/telemetry/posthog_bridge.py` posts batches to the
  documented public batch-capture endpoint `{POSTHOG_HOST}/batch/` with JSON
  `{"api_key": ..., "batch": [...]}` (chosen over the one-event-per-request
  `/capture/` and the SDK-internal `/i/v0/e/` alias). Transport is stdlib
  `urllib.request`, matching `trace.py::_mirror` — `httpx` is only an optional
  extra in `pyproject.toml`, so the core runtime must not import it.
- Events are queued on a bounded in-memory queue (default 2048) drained by a
  single daemon thread that flushes on batch size (default 50) and on a time
  interval (default 2 s), with an `atexit` best-effort final flush bounded by
  a short timeout. When the queue is full, events are dropped and counted
  (`BridgeStats.dropped`); transport failures also drop-with-counter. The
  bridge **never raises into callers**.
- `trace.py` mirrors at span finish: when a `RunTrace` carries `llm.*`
  attributes (see below), `_mirror_llm_span_to_posthog` emits one
  `$ai_generation` event via the bridge. With no API key configured the
  bridge no-ops (single logged notice) and behavior is unchanged.

### Environment variables

| Variable | Meaning |
| --- | --- |
| `POSTHOG_PROJECT_API_KEY` | Project API key; **unset ⇒ the whole mirror cleanly no-ops** with one logged notice. |
| `POSTHOG_HOST` | Ingestion host, default `https://us.i.posthog.com`. |
| `SLM_RUN_ID` | Preferred `distinct_id`; falls back to the hostname, then `"slm-training"`. Runs are machines, not people — no user identity is attached. |
| `SLM_POSTHOG_ENABLE_REPLAY` | Server-side opt-in (`1/true/yes/on`) that sets `posthog.enable_replay: true` in the dashboard feature bootstrap payload. Default off. |

### Event schema

- `$ai_generation` (`capture_ai_generation`): `$ai_trace_id` (the W3C trace
  id), `$ai_model`, `$ai_provider`, `$ai_input_tokens`, `$ai_output_tokens`,
  `$ai_latency` (**seconds**, per PostHog's LLM-analytics convention; callers
  pass milliseconds and the bridge converts), `$ai_is_error`, `$ai_error`,
  `$ai_total_cost_usd`, plus passthrough properties (the trace mirror adds
  `slm.run.id`, `slm.operation`, and any extra `llm.*` attributes).
- `$ai_trace` (`capture_ai_trace`): `$ai_trace_id`, `$ai_span_name`, plus
  passthrough properties.

Span attribute convention introduced by WP-6 (the local OTLP span model had
no prior LLM keys): `llm.provider` / `llm.model` mark a `RunTrace` as an LLM
generation; `llm.input_tokens` / `llm.output_tokens` are optional counts; any
other `llm.*` attribute passes through to PostHog verbatim.

### Dashboard error tracking + replay flag

`src/apps/dashboard/src/features/runtime.ts` already lazily initializes
`posthog-js` when the feature bootstrap selects the PostHog OpenFeature
provider. WP-6 adds `capture_exceptions: true` (uncaught exceptions and
unhandled rejections mirror to PostHog error tracking) and keeps session
recording disabled unless the bootstrap payload carries
`posthog.enable_replay === true` (sourced server-side from
`SLM_POSTHOG_ENABLE_REPLAY` in `src/slm_training/features/runtime.py`). All
existing behavior — lazy imports, OpenFeature provider wiring, in-memory
bootstrap-snapshot fallback — is preserved.

### Free-tier posture

The integration targets PostHog's free tier: low event volume (one
`$ai_generation` per finished LLM-attributed span, occasional `$ai_trace`,
dashboard exceptions), session replay off by default, drop-on-overflow rather
than retry queues, and no server-side PostHog SDK dependency. No scheduled
job, GitHub Action, or CI step talks to PostHog; tests use injected fake
transports and perform no network I/O.
