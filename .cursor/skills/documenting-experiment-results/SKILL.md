---
name: documenting-experiment-results
description: Use when finishing any train, eval, benchmark, profile, telemetry, matrix, or reproduction run — or when a checkpoint was created/promoted without MODEL_CARD/README updates — or when results exist only under outputs/ or in chat without matching docs/design updates
---

# Documenting experiment results

## Overview

**Every experiment updates `docs/design/`.** JSON mirrors plus markdown headlines
are the durable ledger. `outputs/` alone is not enough.

**Every checkpoint also updates `docs/MODEL_CARD.md` and the README model-card
summary.**

## Workflow

1. Identify the doc home (map below).
2. Persist JSON under `docs/design/` (scripts often mirror — verify it matches
   **this** run).
3. Update markdown measured-results: IDs run, pass/fail, recipe (device, steps,
   backend, matrix set, suite `n`, honesty mode).
4. **If a checkpoint was written/promoted/synced:** update
   [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) (roster, eval table,
   history, URI) **and** README → “Model card (summary)”.
5. Do not overclaim (fixture clear ≠ production ship).
6. Commit docs with the experiment.

```dot
digraph docs_after_run {
    "Run finished" [shape=doublecircle];
    "JSON in docs/design?" [shape=diamond];
    "Write/copy JSON" [shape=box];
    "Markdown updated?" [shape=diamond];
    "Update measured-results" [shape=box];
    "Checkpoint created?" [shape=diamond];
    "Update MODEL_CARD + README summary" [shape=box];
    "Caveats recorded?" [shape=diamond];
    "Add recipe / suite-n" [shape=box];
    "Commit with experiment" [shape=doublecircle];

    "Run finished" -> "JSON in docs/design?";
    "JSON in docs/design?" -> "Write/copy JSON" [label="no"];
    "JSON in docs/design?" -> "Markdown updated?" [label="yes"];
    "Write/copy JSON" -> "Markdown updated?";
    "Markdown updated?" -> "Update measured-results" [label="no"];
    "Markdown updated?" -> "Checkpoint created?" [label="yes"];
    "Update measured-results" -> "Checkpoint created?";
    "Checkpoint created?" -> "Update MODEL_CARD + README summary" [label="yes"];
    "Checkpoint created?" -> "Caveats recorded?" [label="no"];
    "Update MODEL_CARD + README summary" -> "Caveats recorded?";
    "Caveats recorded?" -> "Add recipe / suite-n" [label="no"];
    "Caveats recorded?" -> "Commit with experiment" [label="yes"];
    "Add recipe / suite-n" -> "Commit with experiment";
}
```

## Artifact → doc map

| Run / script | JSON (docs) | Markdown |
| --- | --- | --- |
| `run_quality_matrix` | `quality-matrix-results.json` | `quality-experiment-matrix.md` (Vn section) |
| `run_grammar_matrix` | `grammar-matrix-results.json` | `quality-experiment-matrix.md` (X) |
| `reproduce_baseline` | `baseline-reproduction-results.json` | `quality-experiment-matrix.md` |
| `run_phase_pipeline` | `phase-abc-results.json` | `quality-experiment-matrix.md` |
| `run_perf_matrix` | `perf-matrix-results.json` | `perf-experiment-matrix.md` (+ `runtime-performance.md` if latency claim changes) |
| `bench_accel --microbench` | `train-microbench.json` | `runtime-performance.md` / `accel-parallel.md` |
| `bench_telemetry` | `cycle-telemetry.json` | `telemetry.md` if narrative changes |
| `evaluate_model --ship-gates` | summarize scoreboard/gates for the claim | `adversarial-review.md` and/or matrix doc |
| Full HF `train_model` / `remote_train` / `sync_checkpoints` | bucket URI in `checkpoint_bucket.json` | `checkpoint-bucket.md` + **`MODEL_CARD.md` + README summary** |
| `bootstrap_playground` / promoted `*.pt` | — | **`MODEL_CARD.md` + README summary** |
| `profile_generate` | promote if used for claims | perf / runtime docs |

No row? Still add a measured-results note next to the lever's design doc and a
`docs/design/*-results.json` matching existing summary shapes.

## Model card fields (minimum)

In `docs/MODEL_CARD.md`:

- Roster role (demo / matrix champion / production HF ship)
- Run id + path or `hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/`
- Eval table with suite `n` and ship pass/fail
- Recipe: device, steps, context backend, honesty mode
- History row (append; do not erase predecessors)

In `README.md` “Model card (summary)”: one-line table update + link to the
full card. Keep it short.

## Markdown shape (matrix results)

Match existing sections: link the JSON, state host/recipe/`rico_held` n, table
of ID → metrics → ship/perf outcome, honesty caveats. Update in place when
re-running the same IDs; keep invalidated historical rows labeled as such.

## Red flags

- Results only in stdout / PR body
- JSON updated, measured-results table not
- Checkpoint synced but MODEL_CARD / README summary untouched
- "Document after the full matrix"
- Ship pass without suite sizes / honesty mode
- Weakening gates instead of documenting a fail

| Mistake | Fix |
| --- | --- |
| New JSON schema | Extend the existing summary shape |
| Only README long detail | Detail in `MODEL_CARD.md`; README stays a summary |
| Training loss as fidelity | Use parse / `placeholder_fidelity` / struct / reward |
| Skipping failed runs | Document fail + blocker |
