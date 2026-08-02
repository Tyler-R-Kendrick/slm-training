# Autotrain c1793: placeholder-fidelity confirmation rejection

**Verdict:** reject the c1791 placeholder-fidelity candidate and do not enter
promotion. On the exact size-matched fresh-seed confirmation, fidelity weight
1.5 was faster but regressed the declared primary structural metric and the
required binder-reference non-regression metric.

| Arm | Params | Loss / train wall | Smoke quality | p50 | Decision |
| --- | ---: | --- | --- | ---: | --- |
| fidelity 1.5 | 1,608,962 | 23.5152 / 3.60 s | parse 1; meaning .3333; structure .44583; binder F1 .6333; recall .25; fidelity .5278; reward .80333 | 2,344.24 ms | confirmation rejected |
| matched control 0.5 | 1,608,962 | 14.8293 / 2.83 s | parse 1; meaning .3333; structure .45750; binder F1 .9524; recall .25; fidelity .9167; reward .92000 | 5,018.29 ms | matched fresh baseline |

The treatment improved p50 latency by 53.3%, so its MPR/ms efficiency was
higher. That does not confirm a quality champion: structural similarity fell
by .01167, binder-reference F1 fell by .31905, placeholder fidelity fell by
.38889, and reward fell by .11667. Both arms used 23 steps, seed 101793, batch
2, CPU scratch context, and exactly 1,608,962 trainable parameters. Both
completed 3/3 smoke documents without a decode timeout; the AgentV bundles are
present with no execution error.

The run exposed a harness false-positive: Phase A had allowed the efficiency
signal to override the quality-primary and non-regression failures, and the
champion queue consequently wrote `champion_confirmed`. Campaign harness v91
closes that path. Confirmation now requires the policy-owned primary metric
win, rejects required non-regression failures, uses queue resolution as the
only promotion authority, and revalidates previously confirmed queue entries
before promotion selection. Under that corrected policy, c1793 is
`confirmation_rejected:primary_quality_not_reheld`.

This remains fixture evidence (`n=3`). Both arms fail ship gates for evidence
volume, meaningful-program rate, component recall, AST BEQ, canonical BEQ, and
missing held-out/adversarial/OOD/RICO suites. Neither checkpoint is reusable,
promoted, synced, or ship-ready.

Lean is `not_applicable:confirmation_rejected`: no promotion target exists.
The next cycle must first retire the false confirmed queue entry. The next
hypothesis should separate fidelity-weight sensitivity from seed/batch noise
with a new size-matched diagnostic; promotion and formal preflight remain
closed unless a later candidate wins its fresh quality confirmation.

Machine evidence:
[`autotrain-cycle-1793-fidelity-confirmation-rejection.json`](autotrain-cycle-1793-fidelity-confirmation-rejection.json).
