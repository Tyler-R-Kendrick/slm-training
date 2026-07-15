# Tuned accumulation control (2026-07-15)

Linear learning-rate scaling materially changed the result. At the same
2,904-target-token budget, `grad_accum=2` with the default `lr=3e-4` reached
weighted NLL **48.87**, while `grad_accum=2, lr=6e-4` reached **37.22**, close
to the `grad_accum=1, lr=3e-4` baseline at **36.46**. The tuned run persisted
one complete loss-suite evaluation and run insights.

This is a promising recipe candidate, not a default change: it needs multiple
seeds and generated quality-suite validation before promotion.
