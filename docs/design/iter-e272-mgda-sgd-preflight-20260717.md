# E272 — MGDA plus SGD one-step preflight

Date: 2026-07-17
Status: **completed; collinear step improves loss but violates metric-complete guard**

E272 tests the bounded follow-up to E271: retain the 13-active-kind MGDA raw
gradient but apply it with plain SGD, whose update is collinear with the
certified direction. The existing optimizer-state backtracking and held-out
per-kind Pareto guard remain unchanged.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead` at `f016e51`.
The matched one-step preflight took 214.96s.

## Result

MGDA again certified common descent for the active train FTPO-loss objectives
(`min_active_task_dot=4.2178`). SGD improved aggregate held-out loss at every
tested scale; scale 1 changed loss from `2.766007` to `2.765671`. Nevertheless,
all five scales were rejected because nine per-kind guard metrics regressed
across `bind_declaration_root`, `component_bound`, `grammar_comma`,
`grammar_rsqb_root_populated`, `lit`, and `sym`. The regressions affect bad
probability mass, good probability mass, and mean margin.

No step was accepted. The parent was restored with zero held-out delta; the
serialized SHA is
`518d4736571df2f3842ffd338801cfcc4a855d50358c87bd7563facb191935ba`.
Full evaluation matches the parent, five ship gates fail, and AgentV is 2/5.

## Decision

Reject E272 and do not run 30 steps, sync, or promote. The remaining mismatch
is objective completeness, not gradient mixing or optimizer geometry: MGDA
certifies FTPO loss only, while the safety contract independently guards loss,
bad mass, good mass, and mean margin for every decision kind. Even an inactive
loss kind can move in other metrics through shared parameters.

Before further training, profile and solve against the full grammar/AST
metric-gradient constraint set. Do not tune SGD learning rate or duration.

Machine-readable evidence:
[`quality-matrix-v10-e272-one-step-results.json`](quality-matrix-v10-e272-one-step-results.json).
