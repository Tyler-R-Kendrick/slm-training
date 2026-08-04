# Continuous autotrain cycles 1-2, second session (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (this is a second, independent container instance of today's loop; see [summary](continuous-openui-20260801-s2-summary.md)) |
| Campaigns | `continuous-loop-20260801-c1`, `continuous-loop-20260801-c2` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |

## What happened

Both cycles failed before any training/eval metric was produced -- pure
environment bootstrap gaps in a freshly-provisioned container, not lever
findings:

1. **c1** (`c1-control`, `c1-bounds`): `python -m scripts.train_model` raised
   `ModuleNotFoundError: No module named 'torch'`. `torch` is declared as an
   optional `[torch]` extra in `pyproject.toml`, not a base dependency, so a
   bare `pip install -e .` / `uv pip install -e .` leaves it out. Fixed with
   `uv pip install -e ".[torch]"`.
2. **c2** (`c2-control`, `c2-canvas`): training succeeded but
   `scripts.evaluate_model` raised `RuntimeError: AgentV SDK is unavailable;
   run npm ci in the checkout or set AGENTV_RUNNER`. Running `npm ci` itself
   failed first with `node: --import tsx is not allowed in NODE_OPTIONS` --
   this session's `NODE_OPTIONS` is preset to `--import tsx
   --max-old-space-size=8192`, and Node refuses `--import` inside
   `NODE_OPTIONS` during npm's bootstrap phase. Fixed by running
   `env -u NODE_OPTIONS npm ci`.

Both are self-heal per the continuous-loop law ("path/knob/harness failures
are inputs to the next cycle, not reasons to yield"); neither is a harness
bug worth a typed `HarnessSignalV1` since they are container-provisioning
gaps, not code defects. Cycles c3-c5 in this session (see
[c3](continuous-openui-20260801-s2-c3-results.md),
[c4](continuous-openui-20260801-s2-c4-results.md),
[c5](continuous-openui-20260801-s2-c5-results.md)) ran against a working
environment.

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive`
for both cycles (`empty_metrics`, `measurement_incomplete:no_smoke_metrics`,
`primary_metric_unavailable`). Docs-only, local commit.

## Next-run priorities

1. Consider a session-start hook or setup script that runs
   `uv pip install -e ".[torch]"` and `env -u NODE_OPTIONS npm ci` once per
   fresh container, so future continuous-loop sessions don't spend a cycle
   each rediscovering the same two bootstrap gaps.
