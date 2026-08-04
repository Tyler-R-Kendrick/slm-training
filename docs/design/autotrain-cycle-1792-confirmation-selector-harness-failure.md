# Autotrain c1792: confirmation selector harness failure

**Verdict:** harness failure before matrix lock, training, or evaluation. The
driver correctly dequeued c1791 for exact fresh-seed confirmation, then called
the unrelated screening-arm selector. Because every screening family was
already closed or in-flight, the selector failed closed with `registered
screening arm bank exhausted`.

No arm started, no checkpoint was written, no metric was measured, and no
model-quality conclusion is available. The c1791 candidate remains queued;
its attempt must be recovered and replayed with the identical treatment and
control recipes.

Campaign harness v90 makes confirmation and promotion bypass screening
selection. Those paths already carry frozen champion recipes, so choosing a
screening slug is both unnecessary and incorrect. A regression test exercises
the fully exhausted bank: confirmation returns no screening slug while a real
screening cycle still fails closed.

Lean is `not_applicable:harness_failure_before_promotion`: confirmation never
produced a result and no promotion target exists.

Machine evidence:
[`autotrain-cycle-1792-confirmation-selector-harness-failure.json`](autotrain-cycle-1792-confirmation-selector-harness-failure.json).
