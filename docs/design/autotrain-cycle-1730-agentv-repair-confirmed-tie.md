# Autotrain c1730: AgentV repair confirmed; grammar_completion_bounds ties at fixture scale (non-positive)

**Verdict:** the c1729 AgentV SDK repair (`npm ci` at repo root +
`src/apps/openui_bridge` + `src/apps/design_md_bridge`) is confirmed working
end to end. This cycle is the driver's automatic frozen-replay successor to
`continuous-loop-20260802-continuous-openui-202607-98199209-c1` — it reuses the
same size-matched `control`/`bounds` (`grammar_completion_bounds`) checkpoints
and re-runs only evaluation. Both arms now produce a **complete** AgentV
scoreboard (`runner.execution_errors: 0`, full `gates` block) instead of the
`RuntimeError` from c1729. A transient successor campaign
`continuous-loop-20260802-continuous-openui-202607-98199209-c2` hit a CPU wall
timeout (`TimeoutExpired('symmetric decision-arm budget', 155.0)`) before
writing a `cycle_handoff.json` and was correctly skipped by the driver's
recovery logic in favor of this campaign (c3) — a soft timeout, not a repeated
hard blocker, so no action was required.

## Result matrix (complete ship-gate measurement)

| Arm | Records | Parse | Binder F1 | Meaningful | Structure | AST node/edge F1 | p50 | Ship gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1.0 | 0.6333 | 0.0 | 0.0575 | 0.0 / 0.0 | 2,429.53 ms | fail (n, quality) |
| candidate (`grammar_completion_bounds`) | 3/3 | 1.0 | 0.6333 | 0.0 | 0.0575 | 0.0 / 0.0 | 2,455.00 ms | fail (n, quality) |

Ship-gate failures on both arms are the expected fixture-scale failures
(`insufficient_n actual=3 need>=20`, plus quality thresholds below smoke-fixture
reach, plus the four missing ship suites) — not a regression, and not a ship
claim. `grammar_completion_bounds` produces **zero** measurable delta on
`structural_similarity`, `meaningful_program_rate`, or either AST F1 at this
3-record size; latency is a statistical tie (+1.05%, within run-to-run noise).
This lever is a no-op on this fixture at this size, so it is rejected as a
screening candidate for now rather than confirmed either positive or negative.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `fixture_insufficient_n` on both arms plus
`primary_metric_null_or_worse` (0.0 delta) plus `fixture_insufficient_n_alone`.
No stack layer opened (non-positive cycles stay local commits + docs per `sdlc`
autotrain-iteration-delivery). `cycle_intent: retry_measurement`, `climb_state:
rejected`, `ship_state: blocked`.

## Harness feedback

Confirms the c1729 repair closes the AgentV-missing-dependency defect with no
harness code change needed — it was a fresh-container setup gap. Rotate the
next screening arm away from `grammar_completion_bounds` (now measured as a
no-op at this size) to the loss-only `component-plan` supervision arm ranked
first by the driver's next-run priorities.

## Next-run priorities

1. Run the size-matched `component-plan` loss-only screening arm
   (`c20260802-continuous-openui-202607-98199209-c3-component-plan` naming) next.
2. Keep the matched control every cycle; rank on `smoke.structural_similarity`
   with parse-rate/binder-F1 non-regression, per policy v3.
3. Fixture `n=3` insufficient-n failures remain expected diagnostics, never a
   loop terminator; full AgentEvals suites, parameter-efficiency gate, and ship
   gates remain mandatory before any ship claim.

No checkpoint was created or promoted in this cycle — both checkpoints are
reused scratch artifacts from the rejected/inconclusive c1 screening arms. No
model-card or README update required. Machine-readable evidence is in
[`autotrain-cycle-1730-agentv-repair-confirmed-tie.json`](autotrain-cycle-1730-agentv-repair-confirmed-tie.json).
