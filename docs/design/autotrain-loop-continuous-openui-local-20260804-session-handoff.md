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
2. **c3** — `retry_measurement` frozen replay of the c2 checkpoints: AgentV
   now runs, but the smoke suite hit `decode_timeout_count=3/3` on both
   arms.
3. **c4** — a second `retry_measurement` replay reproduced the identical
   timeout shape on both arms.
4. Digging into c3/c4's exact numbers (prompted by review feedback asking
   for the committed per-arm result JSON — see
   [`autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804-results.json`](autotrain-cycle-c3-screening-decode-timeout-host-speed-20260804-results.json)/[`c4...json`](autotrain-cycle-c4-screening-decode-timeout-host-speed-20260804-results.json))
   found the initial diagnosis was wrong: `evaluate_model`'s decode timeout
   is granted per-*chunk*, not per-record, so the real effective budget for
   the 3-record screening chunk was `8s × 3 = 24.0s`, and every arm's actual
   decode wall time landed at 24000–24461ms — a 0–461ms (≤1.9%) miss, not a
   "host is 3x too slow" mismatch. Four same-session arm-measurements at a
   100% incomplete rate, all landing in that same narrow band, crossed the
   locked Pareto policy's own **"High (≫15%) incomplete rate → recalculate"**
   threshold (`autotrain-thrash-timing-pareto-20260803.md`), so this
   iteration applied an evidence-bound recalibration:
   `screening_decode_timeout_seconds` `8 → 10` (`policy.v1.json` `v4 → v5`,
   `harness.autoresearch.experiment_campaign` `v177 → v178`), with a
   regression test pinning real margin over the worst observed sample. Full
   writeup:
   [`autotrain-thrash-timing-pareto-20260804-recalibration.md`](autotrain-thrash-timing-pareto-20260804-recalibration.md).

## State for the next iteration

- Loop id `continuous-openui-local`; last completed campaign
  `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c4`; this
  iteration additionally landed the `screening_decode_timeout_seconds`
  recalibration above (code/policy change, not just docs).
- **Next action:** run a fresh screening cycle (c5) under the recalibrated
  10s budget to confirm it clears the smoke suite on this host — if it does,
  that produces the first real `smoke.structural_similarity` measurement for
  this loop-id/session and unblocks actual model comparison; if it still
  times out, that is new accumulated evidence for a further round.
- No positive **training** result yet this loop-id/session — no stacked PR
  opened for a training win (per `autotrain` continuous-mode SDLC rule:
  stacked PRs only after a positive-result run). The branch PR for this
  session covers the infra bootstrap, the timeout recalibration (code +
  policy + test), and the honest diagnosis docs, including the correction
  above.
- Container bootstrap (`.venv`, torch, `node_modules` incl. `@agentv/core`,
  `openui_bridge`, `design_md_bridge`) is now warm for this container's
  lifetime; a fresh container will need `scripts/setup_dev_env.sh` again.
