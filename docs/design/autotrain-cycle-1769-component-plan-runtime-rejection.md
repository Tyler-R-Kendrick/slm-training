# Autotrain c1769: component-plan is runtime-rejected

**Verdict:** reject the component-plan arm for reproducible candidate-only
runtime failure; quality remains unavailable. Cycle c1769 reused the exact
c1766 checkpoints under the repaired current-main implementation (TwoTower
component v291). No training ran and no new checkpoint was created.

| Arm | Smoke | Decode work | AgentV / gates | Decision |
| --- | --- | --- | --- | --- |
| component-plan | n=3; completed 0; timeouts 3; p50 including incomplete 11,219.33 ms; quality unavailable | 306 forwards; 110,081 states; 110,910 transition misses; 6,334 witness expansions; 115,143 parser forks | bundle complete; no execution errors; quality is not scoreable | **runtime-rejected** |
| matched control | n=3; completed 3; parse 1; meaning .3333; structure .13527; binder F1 .6333; p50 1,146.04 ms | 11 forwards; 31,115 states; 31,277 transition misses; 861 witness expansions; 32,123 parser forks | bundle complete; no execution errors; fail | complete gate rejection |

This cycle tested the c1768 repair hypothesis that bounding the auxiliary plan
bias by its configured weight would prevent trajectory explosion. It did not.
The candidate still selects `Stack` at the root of every smoke document, times
out on all three, and performs effectively the same completion work as c1768.
The magnitude-only repair hypothesis is therefore rejected.

The exact frozen checkpoint has now reproduced its candidate-only timeout while
the matched control completed under both pre-repair and repaired runtimes. That
is enough to reject this arm on runtime behavior without inventing quality
metrics. The supervisor now labels this case `model_runtime`, marks the frozen
arm exhausted, and steers to a distinct registered hypothesis. It no longer
routes another identical replay or another generic harness repair.

The remaining model hypothesis is termination-aware supervision/ranking rather
than cache tuning: state interning sees negligible convergence relative to the
roughly 110,000 unique states, while the model-selected open variadic trajectory
is the source of the amplification. Any future component-plan successor must be
a new preregistered arm with explicit closure/length evidence, not a replay of
this checkpoint.

These are local fixture diagnostics, not production or ship evidence. The
checkpoints remain explicit no-sync artifacts and are not reusable, promoted,
or ship candidates. Lean is `not_applicable:no_champion`; promotion formal
proofs remain mandatory for any future champion.

Machine evidence:
[`autotrain-cycle-1769-component-plan-runtime-rejection.json`](autotrain-cycle-1769-component-plan-runtime-rejection.json).
