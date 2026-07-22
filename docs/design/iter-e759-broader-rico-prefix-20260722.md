# E759 — broader RICO prefix

**Date:** 2026-07-22  
**Decision:** retain model v218; no checkpoint promotion  
**Evidence:** [`iter-e759-broader-rico-prefix-20260722.json`](iter-e759-broader-rico-prefix-20260722.json)

Three matched local CPU replays broadened E758 from RICO n=3 through n=8 and
n=9 without changing any lever. Both n=8 runs and the retained n=9 run score
1.0 for parse, placeholder fidelity, placeholder validity, structural
similarity, tree-edit similarity, component recall, and strict-v2. The n=9
run has reward 0.9337, p50 9068.77 ms, p95 10761.18 ms, zero timeouts, and zero
fallbacks. Every command completed under the 110-second cap.

This validates the shared namespace-capacity invariant beyond the original
three records, but a prefix diagnostic is not a ship evaluation. AgentV is
still 0/1, the required policy suites are absent, and n=9 is below the RICO
minimum. No checkpoint was created or synced. Outputs contain only grammar/AST
symbols, schema enum literals, and declared template markers. The next useful
local probe should sample outside this prefix instead of spending another
cycle on already-green rows.
