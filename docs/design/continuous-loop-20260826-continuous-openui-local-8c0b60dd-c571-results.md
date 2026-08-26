# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c571`

- loop/cycle: `continuous-openui-local` / `571`
- role/intent: `screening` / `retry_measurement`
- recipe: exact frozen c570 pair; local CPU fixture TwoTower, 173 steps, two 70-second arms
- primary metric: `smoke.eval_nll`
- measurement complete: **False**
- result: pre-arm infrastructure failure; no arm launched and no scoreboard or AgentV bundle exists
- failure: symmetric decision-arm preparation had `137.95557885800008s` remaining but required `155.25s` (`17.29442114199992s` short)

The replay persisted all 59 source hypotheses instead of the five executable
members. It also refused the completed control training checkpoint because the
prior *evaluation* recorded 24 decode timeouts, although frozen training reuse
always reruns evaluation. The v291 repair trims retry matrices before
persistence and derives training reuse only from the hash-linked manifest,
completed train summary, and checkpoint. No evaluation result is reused; every
declared evaluation reruns. No gate, arm budget, model capacity, or
constrained-decode invariant changed.

This is failed fixture infrastructure evidence, not an evaluation or ship claim.
