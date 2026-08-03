# Autotrain c1832: confirmation endpoint timeout

**Verdict:** no comparable model result. The repaired symmetric allocator started
both arms, but a promotion-cadence slot incorrectly upgraded the unconfirmed
champion from its smoke screening endpoint to `smoke,held_out`. Both envelopes
timed out before scoreboards; the candidate completed training, while the
control reached only step 2/20.

| Arm | Params | Train | Exposure | Eval | Scoreboard | Disposition |
| --- | ---: | --- | --- | --- | --- | --- |
| tail candidate | 1,608,962 | 20/20, loss 17.6023 | eff=25.81, unique=30/40, repeat=3 | smoke 0/3 complete, 3 decode timeouts; held-out not reached | missing | incomplete |
| matched control | 1,608,962 | 2/20 | unavailable | not reached | missing | incomplete |

The candidate's three smoke documents each received an effective ~8-second
decode allowance and all timed out; its partial eval therefore has no defined
structure, MPR, binder, fidelity, reward, or latency metric. No AgentV bundle,
gate result, or comparison exists. Candidate checkpoint
`a3905259...5cd54` is a local no-sync scratch artifact and is invalid for
serving, promotion, or ship claims. It must not be compared to c1830 because
the endpoint and execution envelope differ.

The terminal matrix correctly emitted `measurement incomplete` and prioritized
an infrastructure retry, but the champion ledger also mislabeled the candidate
`rejected`. Campaign v129 fixes both process gaps: confirmation always remains
on screening endpoints until it succeeds, and an incomplete confirmation is
`confirmation_inconclusive`, never a model rejection. Promotion still requires
a later held-out run and Lean/formal preflight.

Next: block replay of the wrongly escalated frozen manifest, reopen the same
champion, and run a new-seed smoke confirmation under the corrected boundary.
Machine evidence:
[`autotrain-cycle-1832-confirmation-endpoint-timeout.json`](autotrain-cycle-1832-confirmation-endpoint-timeout.json).
