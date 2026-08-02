# Autotrain c1729: AgentV SDK missing-dependency repair (non-positive, incomplete measurement)

**Verdict:** the continuous supervisor's screening cycle
(`continuous-loop-20260802-continuous-openui-202607-98199209-c1`, matched
`control` vs `bounds` grammar-lever arms) trained both size-matched arms to
completion (3/3 smoke documents, zero decode timeouts) but the eval command
exited non-zero for both arms:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

from `src/slm_training/evals/agentv.py:_agentv_runtime`. This is a fresh-checkout
missing-dependency defect, not a model or grammar regression: this container had
never run `npm ci` at the repo root or in `src/apps/openui_bridge` /
`src/apps/design_md_bridge`, so `node_modules/@agentv/core` did not exist and the
AgentV-backed scoreboard step could not resolve `scripts/run_agentv_eval.mjs`.
Per-suite `eval.json` / `eval_smoke.json` were written before the crash (real
`parse_rate`, `structural_similarity`, decode-stat numbers below), but the
driver's SDLC classifier correctly read the campaign as `empty_metrics` on both
arms because the top-level scoreboard (which carries the `evals`/ship-gate
verdict) never got written, and the cycle closed non-positive.

**Repair (self-heal, no code change required):** ran

```bash
npm ci
(cd src/apps/openui_bridge && npm ci)
(cd src/apps/design_md_bridge && npm ci)
```

per the README Quick start. `_agentv_runtime` now resolves
`scripts/run_agentv_eval.mjs` against the repo-root `node_modules/@agentv/core`
(verified interactively). No harness code changed — this was an environment-setup
gap in a fresh container, not a defect in `agentv.py`, so no regression test was
added.

## Result matrix (incomplete — suite metrics only, no AgentV/ship-gate verdict)

| Arm | Records | Params | Steps | Parse | Binder F1 | Meaningful | Structure | p50 | Timeout | AgentV crash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1,608,962 | 21 | 1.0 | 0.6333 | 0.0 | 0.0575 | 2,485.16 ms | 0 | yes |
| candidate (grammar_completion_bounds) | 3/3 | 1,608,962 | 21 | 1.0 | 0.6333 | 0.0 | 0.0575 | 2,217.76 ms | 0 | yes |

`structural_similarity` and every quality metric tie exactly between control and
candidate (delta 0.0) — with `meaningful_program_rate` and `ast_node_f1`/`ast_edge_f1`
at 0.0 for both arms, this fixture's 3-record smoke suite does not exercise the
`grammar_completion_bounds` lever meaningfully at this size, so the tie is
inconclusive rather than a negative result. Latency is 10.8% lower for the
candidate, but latency alone is never the promotion primary and this measurement
is not ship-gate complete (`ship_gates_pass: false` on both arms due to the crashed
AgentV step, independent of the fixture-`n` gate).

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `empty_metrics` on both arms plus
`primary_metric_null_or_worse` (0.0 delta). No stack layer opened; this is
infra self-heal + docs only, per `sdlc` autotrain-iteration-delivery (non-positive
cycles stay local commits + docs, no new stacked PR layer).

## Next-run priorities

1. The queued `retry_measurement` action on
   `continuous-loop-20260802-continuous-openui-202607-98199209-c1` replays the
   identical frozen `control`/`bounds` arms under the now-installed AgentV SDK
   before any new model hypothesis starts — this is automatic per the continuous
   driver's frozen-replay consumption rule.
2. Once the replay produces a complete AgentV-backed scoreboard, re-evaluate
   `grammar_completion_bounds` on the completed ship-gate verdict, not just the
   suite-level tie recorded here.
3. Environment note for future fresh containers running this loop: `npm ci`
   (root + `src/apps/openui_bridge` + `src/apps/design_md_bridge`) and a
   `python -m venv .venv && pip install -e ".[dev,hf]"` are both required before
   the first supervised cycle; neither is auto-provisioned by
   `run_autotrain_continuous`.

No checkpoint was promoted or synced in this cycle; both checkpoints are local
scratch artifacts from a rejected/inconclusive screening arm. No model-card or
README update is required. Machine-readable evidence is in
[`autotrain-cycle-1729-agentv-sdk-repair.json`](autotrain-cycle-1729-agentv-sdk-repair.json).
