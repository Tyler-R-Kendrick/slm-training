# Autotrain c1815: historical claim omission still blocked replay

**Verdict:** measurement incomplete; no model-effect attribution. The direct
v112 claim-copy repair was insufficient because c1815 replayed c1814, whose
experiment artifact had already been written by the older bug with an empty
formal-claim list. The current candidate therefore stopped before evaluation
while its manifest correctly required the fresh proof obligation.

| Arm | Params | Training | Smoke complete | Held-out complete | Current quality | Disposition |
| --- | ---: | --- | ---: | ---: | --- | --- |
| weight 0 control | 1,608,962 | reused c1812 | 0/3 | 0/5 | unavailable: typed decode timeouts | retry frozen pair |
| balance .25 + close 1 | 1,608,962 | reused c1812 | — | — | unavailable: historical claim omission | retry frozen pair |

The current-main Lean preflight proved
`metrics.structural_similarity_monotone` in 1.37 seconds and wrote proof-bound
artifact `65577f5e...ca6a`. The control reproduced the same eight typed decode
timeouts seen in c1812 and c1814. Neither fact supplies the missing current
candidate scoreboard, so the handoff correctly remains `inconclusive`.

Campaign harness v113 recovers an omitted historical claim only from the
predecessor's typed formal-preflight artifact. Recovery verifies the campaign,
experiment, obligation, template, policy, proved status, and recomputed
obligation identity before a fresh successor proof is generated. Missing or
inconsistent artifacts fail closed.

No checkpoint was created or promoted. The next cycle must reuse the same c1812
checkpoint pair and frozen evaluation recipe. If the candidate completes, the
third identical control timeout may be classified against that current result;
otherwise the result remains an infrastructure measurement failure.

Machine evidence:
[`autotrain-cycle-1815-historical-formal-claim-replay-failure.json`](autotrain-cycle-1815-historical-formal-claim-replay-failure.json).
