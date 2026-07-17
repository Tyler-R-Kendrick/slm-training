# E314 visible slot-contract matched train — 2026-07-17

E314 trains the E311 token-pooled component-plan recipe on accepted E314 v2.
The sole data delta is that every train prompt visibly carries its complete
declared slot contract, matching the honest production request shape.

The run stopped at 420 steps / 20,001 target tokens in 125.37 seconds.
Checkpoint SHA:
`f0aaf614e5b6869441c65a091b74429ab9309d508648e51c1c9b4bfcc21a1588`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`. The
scratch prompt vocabulary adds 320 trainable parameters versus E311, so the
capacity comparison is close but not perfectly identical.

| Measure | E311 control | E314 visible contract |
| --- | ---: | ---: |
| Weighted NLL | **4.8819** | 5.0561 |
| Broad NLL | **4.9806** | 5.2258 |
| Final-20 plan loss | **2.3283** | 2.3308 |
| Root accuracy | 0.8500 | 0.8500 |
| Bound top-k recall | 0.4104 | 0.4104 |
| Bound-count MAE | 0.3418 | **0.3416** |

Loss-suite AgentV passes 1/1. Smoke, held-out, adversarial, and OOD exactly
match E311. Limited-RICO structure regresses 0.3333→0.3104. All five held-out
predictions still collapse to `Stack` containing one `TextContent` with the
first visible slot. Seven thresholds fail and AgentV remains 2/5.

**Verdict:** reject the checkpoint while retaining the valid coverage-aware
request transform and v2 corpus. Matching visible slot contracts does not fix
component composition; the next lever must address the decoder's premature
single-child completion directly.
