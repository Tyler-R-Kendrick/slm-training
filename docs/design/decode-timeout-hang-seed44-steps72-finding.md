# Finding: `--decode-timeout-seconds` is not a hard wall-clock guarantee

**Honesty:** `fixture_or_scratch` / smoke, suite `n=1` (isolated record). **Not
a ship claim — a diagnostic finding, not a fix.**

## What was observed

Follow-up on the seed=44 eval exclusion in
[lever-seed-rescue-steps72-measured-results.md](lever-seed-rescue-steps72-measured-results.md):
that eval was killed twice at the `MAX_RUN_MINUTES=3` wrapper (170s, then
178s) with zero stdout either time, against the `lever_seedrescue_s72_seed44`
checkpoint (`twotower`, `steps=72`, `seed=44`, `--asap-decode`).

Isolating by record (`--eval-limit 1 --eval-offset <0|1|2>` against the same
3-record `wf_smoke_test_v1/smoke` suite, `--decode-timeout-seconds 30`,
`--constraint-debt-routing-mode fixed_asap`) against that same checkpoint:

```
offset 0: rc=0  (completes normally, well under 30s decode timeout + AgentV publish)
offset 1: rc=124 (killed at 60s wrapper timeout — 2x the configured 30s decode
                   timeout — zero stdout captured)
offset 2: rc=0  (completes normally)
```

Record offset 0 and 2 completing normally — including a successful AgentV
publish subprocess call each time — rules out the AgentV/node publish step as
the hang source in general. The hang is specific to decoding record index 1
against this particular checkpoint, and it exceeds the configured
`decode_timeout_seconds=30` by at least 2x with no output at all (not even a
partial/errored suite JSON).

## Root-cause hypothesis (not confirmed further this session)

`_generate_chunk` in
[`src/slm_training/harnesses/model_build/eval_runner.py`](../../src/slm_training/harnesses/model_build/eval_runner.py)
(around line 1091) enforces `decode_timeout_seconds` via
`signal.setitimer(signal.ITIMER_REAL, seconds)` + a `SIGALRM` handler that
raises `TimeoutError`. This is a well-known soft limit in CPython: the
interpreter only delivers a pending signal when control returns to Python
bytecode between instructions. If the actual generation call for this record
is blocked inside a C extension call that doesn't periodically release the
GIL / check for signals (a long single `torch` op, a stuck subprocess
`.wait()`/`.communicate()`, or a tight native loop in the grammar-constrained
decode/search path), the alarm fires but is not delivered until that call
returns — which, for this record/checkpoint, apparently never happens within
at least 60s (vs. the configured 30s cutoff).

This was **not** further isolated in this session (no `py-spy`/`faulthandler`
stack dump was captured — this repo's CPU-only sandbox does not have `py-spy`
installed, and adding a new dependency to chase a diagnostic is out of scope
for a docs-only session). This doc records the reproduction, not a fix.

## Why this is being filed as a finding, not a patch

Changing how eval enforces its decode-timeout wall clock is
architecturally significant (it is the harness's SIGALRM-agent safety net,
and the two prior batches in
[the ledger](autotrain-loop-ledger-20260725.md) both quietly excluded similar
killed runs as "not evidence" — this may be the same underlying issue
recurring, not a one-off). It should go through the
`improve-openui-harnesses` skill with a human decision on the right
mechanism (e.g., a `faulthandler.dump_traceback_later` companion trace, a
watchdog thread + `os.kill`, or accepting the soft-limit as a known
constraint and treating "still running past 2x timeout" as its own decode
outcome) rather than being patched blind by a single scheduled session.

## Reproduction

```bash
python -m scripts.evaluate_model \
  --test-dir <wf_smoke_test_v1/smoke fixture, 3 records> --suite smoke \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --device cpu \
  --run-id lever_seedrescue_s72_seed44 \
  --grammar-constrained --decode-timeout-seconds 30 \
  --constraint-debt-routing-mode fixed_asap \
  --run-class scratch_matrix --eval-limit 1 --eval-offset 1
# hangs past 60s with zero stdout (checkpoint from steps=72, seed=44)
```

Checkpoint used: `outputs/runs/lever_seedrescue_s72_seed44/checkpoints/last.pt`
(not committed — `outputs/` is gitignored, scratch-only).

## Next steps

1. Someone with `py-spy`/`faulthandler` available should reproduce and pull a
   stack trace during the hang to name the exact blocking call.
2. If confirmed as a genuine SIGALRM-delivery gap, evaluate a watchdog-thread
   or subprocess-isolation replacement for `decode_timeout_seconds`
   enforcement via `improve-openui-harnesses` — this affects every eval run
   in this repo that relies on the flag, not just this smoke line.
3. Until then, treat any eval killed at `MAX_RUN_MINUTES` with zero stdout as
   a signal this bug may have fired, not just "unlucky decode timeouts,"
   when triaging future killed/excluded runs in the ledger.

Captured: 2026-07-27T15:05:00+00:00
