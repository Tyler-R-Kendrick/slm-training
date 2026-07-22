# E894-E896: low hard-tail replay

E894 continued E886 for 20 CPU steps on E851 base data with 10% requested E872
hard-tail replay. Batch rounding yielded 11.25% effective exposure: 71 base and
9 replay examples. The run completed in 3.26 seconds with final loss 4.05955219
and RMS weight drift 0.00105984. Sync was explicitly disabled.

| Run | Checkpoint / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E893 | E891 / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/1 |
| E896 | E894 / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/1 |
| E892 | E891 / held_out | 5 | 0.8000 | 0.8000 | 0.2000 | 0.8000 | 0.3298 | 0.7143 | 0.7844 | 1 / 0 | 0/1 |
| E895 | E894 / held_out | 5 | 1.0000 | 0.6000 | 0.2000 | 0.6733 | 0.3035 | 0.6476 | 0.8492 | 0 / 3 | 0/1 |

The low-replay arm exactly matches E891's smoke aggregate but is worse on every
held-out semantic/topology metric except parse, reward, and timeout. It loses
0.20 meaning-v1, 0.1267 fidelity, 0.0264 structure, and 0.0667 recall, and
reintroduces three fallback-marked rows. Strict-v2 stays 0.2.

Reject E894 as dominated by E891. Stop replay-ratio tuning and target the
observed typed-role failures (`Form.buttons`, `Form.fields`,
`FormControl.input`, `Tabs.items`, and `SwitchGroup.items`) plus repeated
subtree spam. This diagnostic subset is not ship evidence; AgentV is 0/2 and no
production gate ran.
