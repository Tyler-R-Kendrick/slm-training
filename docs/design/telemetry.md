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
