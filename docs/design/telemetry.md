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

## Meta-trace corpus (G5, SLM-37)

`harnesses/distill/meta_trace.py` formalizes the trace-capture layer for the
future DSL-generating meta-model — schema + retention over artifacts the
stack already writes, per the G5 scope.

**Schema** (`MetaTraceRecord`, pydantic strict/frozen, `schema_version: 1`):
identity (`run_id`, `record_id`, first-class `dsl_id`, W3C `trace_id` /
`traceparent` from the run's `trace.json` sidecar, `source_artifacts`
provenance), request (`prompt`, `slot_contract`), decode spec
(`model_kind`, `checkpoint_sha`, `decode_config`, `seed`,
`deterministic_decode`), outcome (`prediction`, optional `gold`,
`verdicts` — per-example eval metrics joined with the run's honest-gate
pass/failures), and an optional per-step `trajectory` slot for distill
`TraceStore` rows.

**Collector** (`harvest_run_dir`): joins each run directory's existing
`eval_<suite>.json` details, `matrix_result.json`/`scoreboard.json` gate
verdicts, `train_summary.json` recipe, and `trace.json` ids — degrading
gracefully when artifacts are absent. **Retention** (`write_corpus`):
append-only `traces.jsonl` + `manifest.json` with per-line sha256 under a
`campaign.json`-gated tree — the campaign-store conventions, so
`autoresearch.persistence.sync_campaign` can mirror it (dry by default;
local tree authoritative).

**Replay contract** (`replay_trace`): records carry replay-from-spec
identity; bit-exact reproduction is asserted only for deterministic
decoders (`tree_edit_diffusion`'s value-guided search), verified against
the stored `checkpoint_sha` (fail closed on mismatch or on
non-deterministic model kinds). MaskGIT decodes are replay-from-spec, not
bit-exact — stated boundary.
