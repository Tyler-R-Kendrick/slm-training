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
5. **c6** — acked c4's `repair_harness`/`document` actions with the real fix
   commit and let the driver queue another `retry_measurement`. It replayed
   the *same* frozen c2 lineage again and timed out identically — the
   materialized manifest showed the new `decode_timeout_seconds: 10.0`, but
   the actual `evaluate_model` invocation still used `8.0`. This is correct,
   not a bug: `retry_measurement` reproduces the identical frozen arm
   (pinned config, for scientific reproducibility), so it can never exercise
   a policy change — only a **new** hypothesis/matrix does. See
   [`autotrain-cycle-c6-frozen-replay-does-not-pick-up-recalibration-20260804.md`](autotrain-cycle-c6-frozen-replay-does-not-pick-up-recalibration-20260804.md).
   Stopped here rather than acking another identical replay (c7, c8, ...)
   that would reproduce the same 8.0s-pinned result with zero new
   information.

## State for the next iteration

- Loop id `continuous-openui-local`; last completed campaign
  `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c6`; this
  session landed the `screening_decode_timeout_seconds` recalibration
  (code/policy/test, commit `663e2020`) but has **not yet validated it**,
  because every cycle so far has been a `retry_measurement` replay of the
  original (pre-recalibration) c2 lineage, which by design keeps the
  original 8.0s config.
- **Next action:** do **not** ack another `repair_harness`/retry the c2/c3/
  c4/c6 lineage. Start a genuinely **new** screening hypothesis (new
  candidate matrix, fresh `experiment_id` lineage) so the current policy's
  `screening_decode_timeout_seconds=10` is actually used. That run is the
  first real test of the recalibration and, if it clears the smoke suite,
  produces the first real `smoke.structural_similarity` measurement for
  this loop-id/session.
- No positive **training** result yet this loop-id/session — no stacked PR
  opened for a training win (per `autotrain` continuous-mode SDLC rule:
  stacked PRs only after a positive-result run). The branch PR for this
  session covers the infra bootstrap, the timeout recalibration (code +
  policy + test, currently unvalidated), and the honest diagnosis docs,
  including two corrections along the way (the "3x host speed" mischaracterization,
  and the "replay doesn't pick up new policy" clarification).
- Container bootstrap (`.venv`, torch, `node_modules` incl. `@agentv/core`,
  `openui_bridge`, `design_md_bridge`) is now warm for this container's
  lifetime; a fresh container will need `scripts/setup_dev_env.sh` again.
