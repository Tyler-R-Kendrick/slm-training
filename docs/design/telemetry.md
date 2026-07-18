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

## Spans

**Train:** `batch_build`, `forward` → nested `context_encode` + `denoiser_forward`,
`backward`, `optim_step`, `eval_suites`, `device_sync`, `final_save`.

**Generate:** `generate_batch` → `generate_once` / `best_of_n_rank`, plus
`context_encode` inside the model.

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
