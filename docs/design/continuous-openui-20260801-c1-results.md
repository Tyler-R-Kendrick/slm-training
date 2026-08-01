# Continuous autotrain cycle 1 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | Status | Reason |
| --- | --- | --- | --- |
| c1-control | bounds off | **failed** | AgentV SDK unavailable (infra) |
| c1-bounds | `grammar_completion_bounds=true` | **failed** | AgentV SDK unavailable (infra), same class |

No `smoke.*` metrics were produced by either arm — `scripts.evaluate_model`
raised before publishing a scoreboard.

## Diagnostics

1. `src/slm_training/evals/agentv.py::_agentv_runtime` requires
   `node_modules/@agentv/core/package.json` to exist; this fresh sandbox
   checkout had never run `npm ci`, so both arms hard-failed identically at
   eval-publish time (`RuntimeError: AgentV SDK is unavailable; run npm ci in
   the checkout or set AGENTV_RUNNER`).
2. A second, independent blocker exists only in this sandbox's ambient shell:
   `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is set process-wide
   and breaks plain `node` invocations (`node: --import tsx is not allowed in
   NODE_OPTIONS`). Working around it required invoking `npm ci` / the
   continuous driver with `NODE_OPTIONS` unset. This is sandbox-environment
   state, not repo configuration — no code change follows from it.
3. Self-heal: ran `npm ci` (Node deps from `package.json`, including
   `@agentv/core`) and re-ran the driver as cycle 2 with `NODE_OPTIONS`
   unset. Cycle 2's arms completed and published real ship-gate scoreboards,
   confirming the fix and that this was purely an environment-setup gap, not
   a reproducible harness defect (no `HarnessSignalV1` filed).

## Next-run priorities

1. **infrastructure (not harness):** fresh sandbox checkouts must run
   `npm ci` before the first continuous cycle; not a code fix, so no stack
   layer.
2. **model:** re-run bounds vs. control once dependencies are present — done
   as cycle 2 (see
   [`continuous-openui-20260801-c2-results.md`](continuous-openui-20260801-c2-results.md)),
   where the driver's thrash rotation selected the `compact_active_canvas`
   arm instead of a `c1`-identical rematch.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/` (local only, gitignored)
- Runs: `.../runs/c1-control/`, `.../runs/c1-bounds/`
- SDLC delivery record: `outputs/autoresearch/continuous-loop-20260801-c1/sdlc_delivery.json` (`positive=false`, `stack_layer=false`)
- JSON twin: `continuous-openui-20260801-c1-results.json`
