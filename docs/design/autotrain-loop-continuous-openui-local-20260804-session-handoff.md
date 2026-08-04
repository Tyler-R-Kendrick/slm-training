# Continuous autotrain session handoff — 2026-08-04 (continuous-openui-local)

Scheduled-routine iteration summary. Loop `continuous-openui-local`, branch
`claude/great-dirac-u8xw8n`, upstream/integration commit `eba6db30` (no
`origin/main` drift observed this session).

## What happened this session

1. **c2** — first screening cycle on a brand-new ephemeral container:
   `evaluate_model.py --ship-gates` crashed before a scoreboard
   (`RuntimeError: AgentV SDK is unavailable`). Root cause: container had
   never run `scripts/setup_dev_env.sh` (no `.venv`, no torch, no
   `node_modules`). No code defect — bootstrapped the container
   (`pip install -e ".[dev]"`, `npm ci` at repo root +
   `src/apps/openui_bridge` + `src/apps/design_md_bridge`). See
   [`autotrain-cycle-c2-agentv-missing-infra-failure-20260804.json`](autotrain-cycle-c2-agentv-missing-infra-failure-20260804.json)/`.md`.
2. **c3** — replay: training fine, AgentV now runs, but the smoke suite hit
   the per-record decode timeout on all 3 documents
   (`compiler_ms_mean≈23.1–23.4s` vs the locked
   `screening_decode_timeout_seconds=8.0`). Diagnosed as a host-speed
   characteristic of this specific container, not a defect — the 8s default
   was deliberately Pareto-calibrated
   ([`autotrain-thrash-timing-pareto-20260803.md`](autotrain-thrash-timing-pareto-20260803.md))
   against a faster host, and that doc explicitly forbids ad-hoc
   widening from a single failed cycle. See
   [`autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804.md`](autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804.md).
3. **c4** — `retry_measurement` replay of the identical frozen arm
   reproduced the identical timeout (`compiler_ms_mean≈23.1–23.6s`,
   `decode_timeout_count=3/3` both arms) — confirms c3's diagnosis was not a
   one-off cold-start effect; this container's compiler wall time is
   consistently ~3x the locked screening budget.

## Why the loop is pausing here, not spinning on c5/c6/...

A third identical replay of the same frozen manifest would deterministically
reproduce the same timeout again (same steps, same decode budget, same seed)
— not new information, and the locked Pareto policy explicitly rejects
reactive per-cycle wall/timeout widening. Real recalibration needs either:

- **Shrink the recipe** (the Pareto doc's preferred first lever — lower
  `compiler_search_*` cost knobs for the screening role so per-document
  compile time drops under budget on slower hosts), or
- **Accumulated telemetry** across multiple sessions/containers in
  `thrash_timing.jsonl` showing a persistently high incomplete rate before
  an evidence-bound, version-bumped policy change — not two cycles in one
  session on one container.

Both are harness-design work larger than a single soft-failure repair receipt
and are left as the next priority rather than forced through in this
iteration. Per the continuous-mode absolute loop law, timeouts are a soft
failure and never a hard stop — the next scheduled iteration should pick a
**new** screening hypothesis/knob set (not a third identical replay) or take
up the recipe-shrink work above.

## State for the next iteration

- Loop id `continuous-openui-local`; last completed campaign
  `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4`.
- No positive result yet this loop-id/session — no stacked PR opened for
  training results (per `autotrain` continuous-mode SDLC rule: stacked PRs
  only after a positive-result run). The branch PR for this session covers
  the infra bootstrap + honest diagnosis docs only.
- Container bootstrap (`.venv`, torch, `node_modules` incl. `@agentv/core`,
  `openui_bridge`, `design_md_bridge`) is now warm for this container's
  lifetime; a fresh container will need `scripts/setup_dev_env.sh` again.
