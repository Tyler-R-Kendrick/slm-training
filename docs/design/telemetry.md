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
