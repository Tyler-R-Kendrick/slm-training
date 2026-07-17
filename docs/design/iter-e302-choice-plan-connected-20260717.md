# E302 trained component plan plus connected topology (2026-07-17)

E302 composes E301's concise connected-content decoder with the trained E293
choice component-plan checkpoint (SHA-256
`78b70c81bd16395e22718baa91b50427c205f38136269c6248b85562cdec5308`).
The plan head was trained without DESIGN context and runs at decode weight 1.
All other CPU scratch, prompt-only, five-suite settings match E301.

The complete quality board is identical to E301: parse 1.0 on all suites,
meaningful 0.3333/0.0/0.5/0.0/0.6667, seven failed thresholds, and AgentV
2/5 with zero execution errors. The head is not bypassed: it applies 3/5/4/4/35
times across smoke/held/adversarial/OOD/RICO. It changes no choices on the first
four suites and four choices on RICO, without changing aggregate metrics.

**Verdict:** decode weight 1 does not improve component selection. Do not
attribute E301's gain to the trained plan head and do not promote either
checkpoint. A stronger-weight arm can test whether this is merely insufficient
logit scale; if rankings still collapse, training targets must improve.

Artifacts:

- `outputs/runs/e302-choice-plan-connected-close-honest-r1/`
- [machine-readable result](choice-plan-connected-results-iter-e302-20260717.json)
