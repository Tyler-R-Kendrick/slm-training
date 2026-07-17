# E312 token-pooled plan decode scaling — 2026-07-17

E312 re-evaluates the unchanged E311 checkpoint under the frozen E305 honest
policy. The only delta raises component-plan decode weight from 1 to 4. No new
checkpoint is written.

| Suite | Plan changes, weight 1 | Plan changes, weight 4 | Quality change |
| --- | ---: | ---: | --- |
| Smoke | 0/3 | 0/3 | Exact selected-metric match |
| Held-out | 0/5 | 0/5 | Exact selected-metric match |
| Adversarial | 0/4 | 0/4 | Exact selected-metric match |
| OOD | 0/4 | 0/4 | Exact selected-metric match |
| Limited RICO | 1/19 | 4/16 | Structure 0.3333→0.2678; other headline metrics unchanged |

Seven thresholds still fail and AgentV remains 2/5. Parse remains 1.0, but
stronger bias exposes no hidden gain: it changes only limited-RICO decisions
and makes their structure worse.

**Verdict:** reject and stop decode-weight scaling. The learned global
root/count target is misaligned with the decisions needed for held-out and OOD
composition; the next training lever must supervise decision-local semantics.
