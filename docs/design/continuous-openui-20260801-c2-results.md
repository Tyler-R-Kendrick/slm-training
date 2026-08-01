# Continuous autotrain cycle 2 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c2` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | completed | Status |
| --- | --- | ---: | ---: | --- |
| c2-control | canvas off | 3 | 0/3 | **eval_incomplete** — see diagnostics |
| c2-canvas | canvas **on** | 3 | 0/3 | **eval_incomplete** — same cause |

Primary metric unavailable on both arms (`primary_metric_unavailable`,
`measurement_incomplete:no_smoke_metrics`).

## Diagnostics

`evaluate_model.py` raised:

```text
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

Root cause chain, fully diagnosed and repaired in-session:

1. This checkout's `node_modules/` was never installed (`npm ci` had not run),
   so `scripts/run_agentv_eval.mjs` could not `import '@agentv/core'`.
2. Running `npm ci` directly failed with `node: --import tsx is not allowed
   in NODE_OPTIONS` — this session's `NODE_OPTIONS` environment variable is
   malformed (contains literal stray quote characters from the shell
   profile), which breaks bare `node`/`npm` invocations.
3. Fix: `env -u NODE_OPTIONS npm ci` installed `@agentv/core` cleanly.
   Cycles c3/c4 (below) confirm the eval path completes end-to-end afterward.

This is an environment-setup gap in a fresh checkout, not a model or harness
defect. **Non-positive**, no repo code change required — the fix is
session/environment-local (unset `NODE_OPTIONS` before `npm`/`node`).

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive` —
local commit only, no stacked PR for this cycle.

## Next-run priorities

1. No repo change needed; confirmed fixed in-session (see c3/c4 results).
2. Consider a continuous-loop preflight check for
   `node_modules/@agentv/core/package.json` (and a sane `NODE_OPTIONS`) before
   the first AgentV-backed eval, so a bare checkout fails closed with one
   clear message instead of producing empty-metric arm evaluations.

## Artifacts

- Campaign (ephemeral, not committed): `continuous-loop-20260801-c2/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
