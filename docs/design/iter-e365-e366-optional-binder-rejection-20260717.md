# E365–E366 optional-binder rejection — 2026-07-17

E365 corrects the literal interpretation of `decode_min_content`: after two
direct content binders, it temporarily permits either another binder or the
root, and permits a root Stack to select a learned container reference. The
frozen E360 checkpoint is evaluated on RICO rows 0–15 with the retained E350
policy. The run completes in 134.8 seconds.

The opening produces no Cards. Six of 16 rows fall back to a trivial empty
root, meaningful rate falls from 1.0 to 0.625, fidelity from 0.2388 to 0.1013,
structure from 0.2208 to 0.1447, recall from 0.5208 to 0.3125, and reward from
0.7326 to 0.4420. Latency rises to 5.71s p50 / 13.66s p95.

E366 composes the same opening with E360's component-plan decode weight 2.
It completes in 90.0 seconds and exactly reproduces E365's predictions and
quality; plan bias supplies neither Cards nor safe termination. AgentV
correctly reports 0/1 for both diagnostic subsets, and E365/E366 additionally
fail the RICO structure threshold.

Both commands used an external interrupt at 290 seconds and hard kill by 300
seconds. The decoder change was reverted after the negative controls.

**Verdict:** reject the optional-binder opening. A generalized topology path
needs an explicit, visible or learned bounded topology contract; unconstrained
top-level continuation is slower and less accurate.
