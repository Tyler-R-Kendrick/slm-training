# Autotrain c1814: formal-claim replay blocked measurement

**Verdict:** measurement incomplete; do not attribute a model effect. The
frozen promotion replay created a fresh Lean proof and reused the exact c1812
checkpoints, but the candidate stopped before evaluation because its frozen
`formal_claims` were not copied alongside the bound formal obligations. The
control again timed out on every smoke and held-out document. That recurrence
is diagnostic evidence about the control runtime, but it is not a matched-arm
result without a current candidate scoreboard.

| Arm | Training | Smoke complete | Held-out complete | Current quality | Disposition |
| --- | --- | ---: | ---: | --- | --- |
| weight 0 control | reused c1812 | 0/3 | 0/5 | unavailable: typed decode timeouts | retry frozen pair |
| balance .25 + close 1 | reused c1812 | — | — | unavailable: formal replay mismatch | retry frozen pair |

The formal preflight proved `metrics.structural_similarity_monotone` in 1.39
seconds and bound the current proof to SHA-256 `d8111368...bbe`. This proves the
registered metric obligation only. It does not prove candidate quality.

The harness failure exposed two process gaps. Frozen promotion replay must copy
the experiment's formal claims as well as the campaign obligations, and a
runtime-unblock classification must require usable current candidate metrics.
The repair now enforces both conditions and rewrites the derived c1814 handoff
to `inconclusive` with an exact frozen retry.

No checkpoint was created or promoted in this cycle. The next run must keep the
same checkpoints, suites, gates, proof template, and frozen experiment identity.
Only after both current scoreboards exist may the repeated control timeout be
classified terminally.

Machine evidence:
[`autotrain-cycle-1814-formal-claim-replay-harness-failure.json`](autotrain-cycle-1814-formal-claim-replay-harness-failure.json).
