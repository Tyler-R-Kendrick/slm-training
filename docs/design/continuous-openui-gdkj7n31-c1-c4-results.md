# Continuous autotrain loop `gdkj7n31`, cycles c1-c4 (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `gdkj7n31` |
| Upstream / integration commit | `41d874c76b9ed68f4c6d375366ea4398b95a0429` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes / arm |
| Driver | `python -m scripts.run_autotrain_continuous --loop-id gdkj7n31 --train-version wf_smoke_v2 --steps 20` |

## Run matrix

| Cycle | Arm | smoke n | parse_rate | mpr | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c1 | control | — | — | — | — | **harness_failure** — AgentV SDK unavailable (empty_metrics) |
| c1 | bounds | — | — | — | — | **harness_failure** — same |
| c2 | control | — | — | — | — | **harness_failure** — AgentV SDK unavailable (empty_metrics) |
| c2 | canvas | — | — | — | — | **harness_failure** — same |
| c3 | control | 3 | 1.0 | 0.0 | 7323.04 | eval completed; ship gates fail (insufficient n) |
| c3 | both (bounds+canvas) | 3 | 1.0 | 0.0 | 7491.72 | eval completed; ship gates fail (same) |
| c4 | control | 3 | 1.0 | 0.0 | held_out.structural_similarity=0.38248 | eval completed; ship gates fail (insufficient n) |
| c4 | steps | 3 | 1.0 | 0.0 | held_out.structural_similarity=0.37006 | eval completed; ship gates fail (same); worse than control |

All four cycles classified **non-positive** (SDLC Phase A) — no stack layer opened for any of them.

## Diagnostics

1. **c1-c2 root cause (harness_failure, not model quality):** the session's Python
   venv had no Node dependencies installed yet — neither the repo-root nor
   `src/apps/{openui_bridge,design_md_bridge}` had run `npm ci` — and the
   sandbox injects a global `NODE_OPTIONS="--import tsx" --max-old-space-size=8192`
   that this Node build rejects outright (`node: --import tsx is not allowed in
   NODE_OPTIONS`). `publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`)
   spawns `node` without sanitizing the inherited environment, so both arms in
   c1 and c2 failed closed with `AgentV SDK is unavailable` before producing any
   scoreboard (`empty_metrics`).
2. **Self-heal (session-local, no tracked-file change):** ran `npm ci` at repo
   root and in both `src/apps` bridges, and cleared `NODE_OPTIONS` before
   invoking the continuous driver for c3-c4. This is *executable unblocking*
   (definition 3 in `autotrain-iteration-delivery.md`) — the identical
   `wf_smoke_v2` / `e938_role_safe_all_targets_v2` recipe then produced real
   scoreboards. Nothing was committed for this fix because it changed no
   tracked file (only local venv/`node_modules` state); the equivalent
   **code-level** fix — stripping `NODE_OPTIONS` before spawning the AgentV
   `node` subprocess, mirroring the existing `_sanitized_env()` pattern in
   `src/slm_training/dsl/grammar/backends/graphql_js.py` — already exists as an
   open, unmerged PR from a prior session:
   [#1254](https://github.com/Tyler-R-Kendrick/slm-training/pull/1254)
   (`fix(ci): resolve E741 breaking main CI + sanitize AgentV NODE_OPTIONS`,
   branch `claude/great-dirac-wcu1ad`). Not duplicated here; recommend merging
   it so future sessions skip this re-diagnosis.
3. **c3-c4 (real, honest results):** post-heal, both cycles ran clean but show
   null-or-worse primary-metric deltas at fixture `n=3` (`fixture_insufficient_n`
   ship-gate fail is expected at this scale, not a regression). c4's diagnosis
   signal: "experiment improved locally but still fails honest ship gates;
   continue SFT/architecture experiments; keep RL locked."

## Next-run priorities

1. **infrastructure:** merge PR #1254 so the AgentV `NODE_OPTIONS` sanitization
   ships and this env gap stops recurring across sessions.
2. **model:** continue SFT/architecture experiments at larger step budgets;
   keep RL locked per the c4 diagnosis.
3. **evaluation:** re-test `bounds`/`canvas`/`steps` levers at matched or larger
   `n` once a session's wall budget exceeds fixture smoke scale; keep ship
   gates honest, do not weaken for continuous smoke.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c{1,2,3,4}/`
  (gitignored; not part of this PR)
- Per-cycle summaries: `.../measured-results-continuous.md` in each campaign dir
- JSON twin: `continuous-openui-gdkj7n31-c1-c4-results.json`
