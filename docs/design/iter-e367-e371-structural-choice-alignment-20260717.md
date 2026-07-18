# E367–E371 structural choice alignment — 2026-07-17

The production choice tokenizer trains on a marker-free structural stream:
top-level expressions are emitted in dependency order and the final expression
is the root. The earlier minimum-content decoder instead forced the separate
`=` / `r=` compatibility codec, so it could not express a learned intermediary
Card binder.

E367 switches the minimum-content path to the trained structural codec on the
frozen E360 checkpoint. Its unrestricted arrays admit arbitrary nested values:
parse falls to 0.6875 and no Card survives canonicalization. This establishes
that codec alignment alone is insufficient.

E368 resumes E360's full state on the immutable E357 Card-hierarchy corpus. The
CPU run is explicitly capped at 4.5 minutes and stops on the wall-time budget
after 270.4 seconds, 293 cumulative steps, and 15,091 target tokens. The final
local checkpoint SHA is
`a7dd2d79df0c3c36f80e1bb81349f7dad663d9796d201ffff492535a11982573`.
The run does not reach its 18,000-token target, performs no final loss eval,
does not improve the inherited best NLL of 5.8091, and is not synced.

E369 adds plan-guided structural termination, but a tokenizer-contract bug
classifies choice references such as `&0` as `bind` while the new filter looks
for `ref`. The filter therefore fails open, permitting empty arrays. All 16
outputs are trivial and AgentV correctly reports 0/1.

E370 is a three-row remediated-suite smoke after replacing the kind lookup with
the codec's actual `&` reference syntax. E371 is the intended frozen first-16
RICO diagnostic. The corrected raw stream now creates nonempty Card and Stack
edges. Relative to E369, E371 restores parse from 1.0 to 1.0, meaningful rate
from 0 to 0.1875, fidelity from 0.0063 to 0.1257, structure from 0.0894 to
0.1405, recall from 0 to 0.1667, and reward from 0 to 0.1357. It remains below
the retained E350 policy and fails AgentV 0/1 because it is a 16/1500
diagnostic and structure is below 0.2.

Every command used an external 290-second interrupt with a forced kill ten
seconds later. E368 additionally used the internal 4.5-minute training budget.

**Verdict:** retain the generalized reference-kind fix and its codec-state
regression test, but reject E368/E371 for promotion. Structural choice decoding
now expresses a Card hierarchy correctly; learned topology count and coverage
remain the next bottleneck.
