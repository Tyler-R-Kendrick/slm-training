# E265 — Pareto-safe gold-AST set FTPO updates

Date: 2026-07-17
Status: **completed; aggregate exact-state guard improved, semantic quality regressed; checkpoint rejected**

E265 moves the E264 held-out guard into the optimizer. Each 30-step
`ftpo_set` proposal starts from the last accepted model and is evaluated on all
39 held-out E261 exact states. A proposal is accepted only if held-out loss and
bad-token mass do not increase, good-token mass and mean margin do not
decrease, and at least one metric improves. Failed proposals are retried from
the same model and Adam state at scales `1, 1/2, 1/4, 1/8, 1/16`; if every
scale fails, both model and optimizer state are restored.

Immediately before training, the branch fetched and rebased latest
`origin/main`, was clean, and proved `0 behind / 1 ahead` at harness commit
`d2c295aabda7ac9a33d7e569b8dfabfa8f5ceba7`. The run used the unchanged E228
parent, 200 committed E261 train events, 39 held-out events, CPU HF context,
learning rate `5e-5`, and eight-step compiler-tree evaluation. Training trace:
`818698d3bfe50ad9856670867cf9d91b`. Evaluation trace:
`6ab324f4872a1f56cde94d539671a8b8`.

## Safe-update result

Only 3 of 30 proposed updates were accepted:

| Step | Accepted scale | Held-out loss | Bad mass | Good mass | Mean margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | parent | 2.7660 | 0.039225 | 0.041869 | 1.0035 |
| 1 | 0.5 | 2.7294 | 0.038862 | 0.042637 | 1.0998 |
| 14 | 0.25 | 2.7247 | 0.038829 | 0.043244 | 1.1172 |
| 15 | 0.5 | 2.7189 | 0.038805 | 0.044438 | 1.1487 |

The other 27 updates were rejected. Across the run, 142 candidate scales
required 5,538 held-out event forwards and 3,009 seconds (50m09s) for the local
preference stage. This implementation is correct but too expensive for routine
matrix use without batched/cached validation.

The final checkpoint genuinely differs from E228: 98 of 374 tensors changed,
with maximum absolute delta `6.32e-5` and aggregate L2 delta `0.03322`.
Checkpoint SHA is
`44079a8ca0f52a29034794dd5ddef0637b37d57a21b0795b1486a3c2dda846ab`.

## Aggregate guard hid decision-kind regressions

The aggregate guard passed, but it permitted gains in some decision kinds to
mask losses in others. Notable held-out regressions include:

- `grammar_comma`: loss `1.3764→3.1417`, bad mass `0.00331→0.00750`, good
  mass `0.01418→0.00610`, and mean margin `0.5611→-1.1086`;
- `bind_reference_bound_children`: loss `2.6412→2.6805`, bad mass
  `0.07127→0.08885`, and mean margin `0.2271→0.1654`;
- smaller good-mass or margin regressions in component, literal, symbol, and
  closing-bracket decisions.

Therefore aggregate exact-state dominance is not a sufficient safety contract.
The next guard must enforce the same grammar/AST-derived metrics within each
decision kind, so high-volume or easy categories cannot compensate for a
semantic category regression.

## Full evaluation

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | Reward vs current parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.8073 | -0.0703 |
| held_out | 5 | 1.0000 | 0 | 0.3533 | 0.4044 | 0.7646 | -0.0668 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.6667 | 0.5355 | 0.8550 | -0.0560 |
| ood | 4 | 1.0000 | 0 | 0.2583 | 0.3750 | 0.7265 | -0.0895 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.2500 | 0.2580 | 0.7480 | +0.0120 |

Meaningful-program rates and component recall are unchanged from the current
parent control. Fidelity drops on four suites; reward drops on four and rises
only on `rico_held`. Five ship thresholds fail and AgentV passes 2/5 with zero
execution errors. Syntax remains 1.0 with zero fallback and timeout counts,
which continues to validate the deterministic compiler-tree path rather than
learned structure.

## Decision

Reject and do not promote or sync E265. Retain the generalized
optimizer-consistent reject/backtrack mechanism as harness evidence, but do not
run it at scale until validation is batched/cached. The optimization hypothesis
is only partially confirmed: safe aggregate exact-state progress exists, but
the aggregate guard is semantically under-specified. The next experiment must
use a decision-kind-stratified grammar/AST guard, not another literal case rule
or a longer run.

Machine-readable evidence:
[`quality-matrix-v10-e265-results.json`](quality-matrix-v10-e265-results.json).
