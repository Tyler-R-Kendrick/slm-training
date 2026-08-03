# Continuous autotrain: 2026-08-03 cycle 4, session n8vwtq (blocked — corrected diagnosis)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c4`
**Intent:** `retry_measurement` (3rd consecutive attempt at the same frozen
component-plan-vs-control comparison)

**Result: both arms timed out again** (`measurement_incomplete` for both
`c4-control` and `c4-component-plan`). This is the 3rd consecutive cycle
(c2, c3, c4) failing on the identical measurement with the same signature —
per `continuous.md` rule 4, this is now a **repeated hard blocker**, not a
soft timeout to keep silently retrying.

## Correction to cycles c2/c3's diagnosis

The harness fix in commit `071560ee9da5918f6b15ff71b7c6b4a66bb7265f` and its
accompanying docs (`continuous-openui-local-n8vwtq-c2-results.md`,
`-c3-results.md`) claimed the `control` arm trains with
`compiler_decode_mode="off"` and only the `component-plan` arm uses `"tree"`,
and that reallocating the eval wall-budget by that knob would fix the
timeout. **This premise is wrong**, verified directly against the on-disk
experiment/scoreboard artifacts for this exact hypothesis family:

- `docs/design/continuous-openui-local-n8vwtq-c{2,3,4}-results.json`'s own
  `control`/`component-plan` experiment knobs both show
  `"compiler_decode_mode": "tree"` — they are **not** different.
- Ship-gate evaluation always runs under `evaluation_policy.compiler_decode_mode
  = "tree"` (`strict_compiler_tree`, the honest slot-contract policy) **regardless**
  of any training-time knob — confirmed on cycle 1's control scoreboard, which
  trained with `--compiler-decode-mode off` yet still evaluated under `"tree"`.

The real, much better-supported explanation for the latency difference
between cycle 1 (fast: control `latency_ms_p50=1685ms`, `structural_similarity
=0.0575`, near-degenerate output) and cycles 2-4 (slow: control
`latency_ms_p50=17781ms`, `structural_similarity=0.327`, genuinely structured
output) is **model-output complexity**, not a static per-arm knob: a
better-trained model that emits valid, deeper component structure requires
the constrained tree decoder to search a much larger legal-candidate space
per token than a near-degenerate model does. That is not predictable from a
knob ahead of time, so a fixed decode-mode-based budget reallocation cannot
fix it.

Commit `071560ee` is **not reverted** — the `_fit_decode_weighted_arm_budgets`
helper is a reasonable, tested safety net for the (different, real) case
where two arms in a comparison genuinely use different decode modes, and it
correctly degrades to a symmetric split when modes match (which is exactly
what happened here — both arms got the same allocation, as the regression
test `test_decode_weighted_arm_budget_is_symmetric_when_modes_match` predicts).
But it does **not** address this cycle's actual blocker, and the c2/c3 docs'
claim that it would is corrected here.

## Disposition: blocked (host speed), not a harness or model defect

This specific host's CPU cannot complete a 2-arm, 3-document, honest
`strict_compiler_tree` ship-gate comparison for this richer-output hypothesis
within the repository's `MAX_RUN_MINUTES=3` hard cap — regardless of how the
budget is split between arms, because **both** arms need comparable real
decode time once the model is genuinely trained. Per repo law, `MAX_RUN_MINUTES`
is not weakened to force this through. This is not scored as a negative model
result: the hypothesis is already independently reproduced positive 3 times
on other (faster) hosts (`#1369`, `#1376`, `#1378`); this host simply cannot
independently re-confirm it under the honest ship-gate suite within budget.

No stacked PR for this cycle (blocked, not positive). Per `continuous.md`,
the loop should route to a *different*, lighter-weight hypothesis next
(e.g. `component-token` or `slot-component-coverage`, both proposed in this
cycle's matrix) rather than a 4th identical retry of the same blocked arm.

Machine evidence: same schema as prior cycles; no new JSON emitted for this
blocked cycle beyond this correction note, since both arms produced no
scoreboard (nothing new to tabulate beyond "timed out again, 3rd time").
