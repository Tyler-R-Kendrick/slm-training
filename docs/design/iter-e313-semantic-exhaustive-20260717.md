# E313 semantic-exhaustive choice alignment — 2026-07-17

Status: completed; checkpoint rejected; not promotable or ship.

E313 adds the existing semantic-exhaustive compiler-alignment loss to E311's
matched E307 v4 / CPU scratch / 20k-target-token recipe. It trains the actual
denoiser logits at gold compiler-legal root and bound decisions without adding
model parameters.

The first launch stopped after step 7 / 336 target tokens. A gold alignment
token was absent from the compiler decision's candidate tuple, and the harness
raised `ValueError: tuple.index(x): x not in tuple`. No checkpoint was written,
so there is no model-card or bucket update. The last completed batch had 20
alignment rows (2 root, 5 bound) and alignment loss 22.1860.

The harness now skips and counts those invalid rows. A regression test verifies
that alignment remains finite, and r2 completed the unchanged recipe at 420
steps / 20,001 target tokens in 199.42 seconds. It aligned 6,746 rows and
skipped 109 gold-outside-candidate rows. Alignment loss learned strongly:
24.6782 on the first batch, 3.6404 on the last, and 2.6819 over the final 20.

Checkpoint SHA:
`3495bb22c1472c830f317cce9706dfadb7558b0c9e6139cb6436dbba75a32781`.
It is a local scratch artifact with explicit `--no-sync-checkpoints`.

| Measure | E311 control | E313 semantic alignment |
| --- | ---: | ---: |
| Weighted NLL | **4.8819** | 5.0604 |
| Broad NLL | **4.9806** | 5.1577 |
| Final-20 plan loss | 2.3283 | **2.0900** |
| Root accuracy | 0.8500 | **0.9000** |
| Bound top-k recall | 0.4104 | **0.4292** |
| Bound-count MAE | 0.3418 | **0.3344** |

Loss-suite AgentV passes 1/1. Under the unchanged honest policy, smoke,
held-out, adversarial, and OOD exactly match E311. Limited-RICO structure
regresses 0.3333→0.3278 while its other headline metrics are unchanged. Seven
thresholds fail and AgentV remains 2/5.

**Verdict:** reject E313. Direct semantic decision loss and auxiliary plan
metrics learn, but neither causes held-out/OOD generation gains. Stop this
auxiliary-supervision family; the next lever should change the task/data
formulation that supplies decision-local prompt semantics.
