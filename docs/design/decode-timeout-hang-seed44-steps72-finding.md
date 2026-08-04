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

## Non-reproduction attempt with `py-spy` (2026-07-28, scheduled autotrain-loop session)

**Honesty:** `fixture_or_scratch` / smoke, `n=3` isolated attempts. **Not a
fix — a failed reproduction, recorded per this repo's iron law (a negative
result is still evidence).**

Acted on next step 1 above. `py-spy 0.4.2` was installed (`pip install
py-spy`, no `py-spy`/`faulthandler` dependency existed in this fresh
session's checkout) specifically to attach during the hang and capture a
stack trace. Recipe:

1. Rebuilt the eval fixture: `python -m scripts.build_test_data --source
   fixture --no-rico-path --train-manifest
   src/slm_training/resources/data/train/wf_smoke_v2/manifest.json --suites
   smoke` (fresh 3-record `smoke` suite, `outputs/data/eval/v1`).
2. Retrained a fresh `seed=44, steps=72` checkpoint with the exact recipe
   from
   [lever-seed-rescue-steps72-measured-results.md](lever-seed-rescue-steps72-measured-results.md)
   (`run-id lever_seedrescue_s72_seed44_repro`) — real run, 16.4s wall,
   completed normally.
3. Ran the offset=1 reproduction command from this doc three times against
   that fresh checkpoint, backgrounding the first attempt specifically to
   attach `py-spy dump --pid <pid>` once past the 30s decode timeout.

**Result: the hang did not reproduce in any of 3 attempts.**

- Attempt 1 (backgrounded, `py-spy` armed at +32s): the process had already
  exited by the time `py-spy` attached (`Failed to get process executable
  name` — no such process). It ran to completion and failed at the AgentV
  publish step (`RuntimeError: AgentV SDK is unavailable; run npm ci` — this
  session never ran `npm ci`, an expected, unrelated environment gap, not
  the hang).
- Attempts 2 and 3 (foreground, timed): `rc=1`, wall-clock **31.17s** and
  **31.74s** respectively — both land just past the configured
  `--decode-timeout-seconds 30`, consistent with the ASAP-fallback path
  engaging at the timeout boundary, then the same AgentV/`npm ci` failure.
  Neither attempt hung past ~32s or produced zero stdout; both produced a
  full Python traceback ending at the publish step.

This is the opposite of the original observation (killed twice at 170s/178s
with **zero stdout**, i.e. >2x the 30s timeout with no output at all). Three
consecutive non-hangs against a freshly retrained (same seed, same recipe,
but not byte-identical — different session, different `.venv`) checkpoint
does not rule out the original hang; it narrows what's *not* sufficient to
reproduce it:

- Not purely "record offset 1 against a `steps=72, seed=44` checkpoint" in
  general — this session's independently retrained checkpoint with those
  exact settings decodes that record in ~31s, not >60s, three times running.
- Not the AgentV/node publish subprocess — that step is reached and fails
  fast (immediately, on missing `npm ci`) in all 3 attempts here, so it
  cannot itself be a multi-minute stall source in this environment.

**What remains open:** either (a) the hang is genuinely intermittent /
input-dependent on something not controlled by `--seed` alone (checkpoint
weight bytes can differ run-to-run even at fixed seed across different
`torch` builds or thread counts — see the still-unresolved `s36_seed42`
cross-session discrepancy in
[lever-seed-rescue-steps72-measured-results.md](lever-seed-rescue-steps72-measured-results.md)),
or (b) it depended on the *original* session's specific checkpoint bytes or
host load and this session's retrained checkpoint is a different object that
happens not to trigger it. Both are still open; this session did not settle
which. The original checkpoint (`outputs/runs/lever_seedrescue_s72_seed44/`)
was never committed (`outputs/` is gitignored) and no longer exists to
re-test directly.

**Revised next steps:**

1. The `py-spy`-armed reproduction harness above is now cheap to re-run
   (checkpoint trains in ~16s, `py-spy` is installable via `pip`) — if the
   hang recurs in any future session, attach `py-spy dump` (already proven
   to work under this sandbox's root/no-ptrace-restriction setup) before the
   process is killed, rather than only recording a zero-stdout kill.
2. Do not treat 3 non-hangs as "fixed" or "not real" — the original
   170s/178s zero-stdout kill is real, measured evidence per this repo's
   iron law, and still stands. This finding only shows non-reproduction with
   a different checkpoint instance, not a root cause.
3. Diagnosing the checkpoint-byte-level cross-session discrepancy (open
   next-step 2 in
   [lever-seed-rescue-steps72-measured-results.md](lever-seed-rescue-steps72-measured-results.md))
   would likely explain both this non-reproduction and the `s36_seed42`
   discrepancy at once, since both point at the same unconfirmed candidate
   causes (floating-point non-determinism, `torch` build/version drift, or
   CPU thread-count differences across sessions).

Captured: 2026-07-28T02:35:00+00:00
