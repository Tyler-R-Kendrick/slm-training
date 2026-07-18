# VSS1-04 solver trace + replay — fixture evidence (2026-07-18)

Fixture-grade wiring evidence for SLM-64 (VSS1-04): making certified-solver
transitions replayable and measured on the **existing** decode trace + telemetry
owners. No second trace store, no new output root, no custom binary format.

## What was implemented

- [`dsl/solver/replay.py`](../../src/slm_training/dsl/solver/replay.py) — Torch-free
  typed event builders (`solver_state`, `support_result`, `certified_deduction`,
  `decision`, `backtrack`, `nogood`, `solver_terminal`), mode-gated certificate
  serialization (`none` / `summary` / `full`), and `solver_replay_violations`
  (ten invariants, human-readable strings).
- [`harnesses/distill/trace_store.py`](../../src/slm_training/harnesses/distill/trace_store.py)
  — decode trace **schema `version = 3`** (backward-compatible v1/v2 readers),
  `DecodeTraceRecorder.record_solver` sidecar, and `replay_violations` extended to
  run the solver invariants on any solver events present.
- [`models/decode_stats.py`](../../src/slm_training/models/decode_stats.py) — solver
  work-metric counters + `solver_ms` (separated from denoiser/projection), zero on
  the default path, surfaced only under `metrics["decode_stats"]` via
  `aggregate_stats`.
- [`models/twotower.py`](../../src/slm_training/models/twotower.py)
  `_record_solver_metrics` — folds closure counters into `DecodeStats` and, when a
  `DecodeTraceRecorder` is attached, emits the closure-subset events + a bounded
  certificate/counter sidecar. No-op when neither stats nor a recorder is active.

## Schema version

- Decode trace: `TRACE_VERSION = 3` (was 2). v1/v2 rows load and replay unchanged.
- Solver event stream: `SOLVER_TRACE_SCHEMA_VERSION = 1`.
- Certificate schema: `CERTIFICATE_SCHEMA_VERSION = 1` (unchanged; VSS0-04).

## Privacy / boundedness

Events and certificates carry only token/path ids and SHA-256 digests — never raw
region/user text (the terminal verifier-report summarizer drops non-allowlisted
strings). The `solver_state` domain snapshot is bounded; a truncated snapshot sets
`trace_truncated=true` and the validator reports the trace non-replayable rather
than accepting bounded evidence as an exhaustive proof.

## Test command

```bash
python -m pytest \
  tests/test_dsl/test_solver_replay.py \
  tests/test_harnesses/distill/test_solver_trace.py \
  tests/test_models/test_decode_stats.py \
  tests/test_models/test_trace_store.py \
  tests/test_harnesses/distill/test_meta_traces.py -q
python -m scripts.repo_policy
```

Result: solver-replay + trace + decode-stats + historical-compat suites pass;
`repo_policy` ok. (The issue's suggested `tests/test_runtime` path does not hold
trace tests — the runtime-trace tests are `tests/test_runtime_trace.py`; the
decode-trace/replay tests are the paths above.)

## Honesty

Fixture-grade wiring only: the event schema, the ten replay invariants (including
`full`-mode certificate tamper detection), the decode-stats counters, and
historical-trace compatibility are tested on tiny closed fixtures. No model,
checkpoint, training corpus, or eval run is produced or claimed. Closure never
reports `solved`. **No solver-quality, correctness, speed, or ship claim is
made.**
