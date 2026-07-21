# E652 — display-value text role

Date: 2026-07-20
Status: completed negative scratch result; reverted; not ship

E652 mapped display `value` roles to text-compatible properties so metric
values could receive simple visible leaf owners. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E650 policy. It
completed without timeout/fallback and emitted AgentEvals plus AgentV evidence.

| OOD `n=4` | E650 | E652 |
| --- | ---: | ---: |
| meaningful v1 / strict v2 | 1.0000 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / recall | 0.7355 / 0.8750 | 0.7824 / 0.8750 |
| reward | 0.9790 | 0.9798 |
| node / edge F1 | 0.7987 / 0.5798 | 0.8410 / 0.5893 |
| latency p50 / p95 | 2844.86 / 7879.73 ms | 2932.12 / 6767.31 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Dashboard emits two clean metric TextContent leaves, but they are detached root
sections; both planned Cards are empty and one metric is duplicated into the
Callout. Meaningful v1 correctly flags a trivial layout and strict v2 is
unchanged. Reject the treatment stamped v99. After rebasing onto E651 restoration
v101, the append-only lineage records E652 as treatment v102 and restoration
v103. The next mechanism must
bind each metric leaf inside its specific Card. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e652-value-text-role-20260720.json).
