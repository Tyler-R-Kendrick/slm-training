# Autotrain cycle 2 — component-plan screen

The paired CPU fixture screen produced a structural-similarity win, but it is
not yet evidence that the model meaningfully learned OpenUI programs.

| Arm | params | parse | meaningful | structure | component recall | p50 request latency | gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 1,755,764 | 1.000 | 0.000 | 0.3267 | 0.1667 | 23,124.96 ms | fail |
| Component-plan candidate | 1,755,764 | 1.000 | 0.000 | **0.3828** | 0.1667 | **18,272.11 ms** | fail |

The paired primary delta is `+0.05613` and both arms completed without runtime
errors, so this is a positive fixture screen and the candidate is queued for a
fresh-seed confirmation. It is not a champion: smoke `n=3` is below the `n>=20`
minimum, all full suites are absent, meaningful-program/AST/canonical/fidelity/
reward metrics remain zero, and the candidate is only a structural result.
Both arms are size-matched at 1,755,764 trainable parameters; no capacity claim
is made. Checkpoints remain local scratch artifacts with explicit no-sync.

Latency now reports observed batch-wall request completion separately from
amortized throughput (`6090.70` vs `7708.32` ms/record for the candidate/control
medians). Cache replays cannot supply these runtime measurements.

Lean remains a formal applicability gate, not a learning signal. The canonical
formal tests pass in the mocked environment, but live `lake` is unavailable on
this host, so no proof success is claimed. Next run: fresh-seed confirmation of
component-plan with the exact matched control; keep promotion formal preflight
locked until that confirmation and meaningful-program evidence exist.

JSON twin: [autotrain-cycle-20260803-c2-component-plan-screen.json](autotrain-cycle-20260803-c2-component-plan-screen.json)
