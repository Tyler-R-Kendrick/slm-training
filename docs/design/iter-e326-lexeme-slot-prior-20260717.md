# E326 corpus-derived slot-lexeme prior — 2026-07-17

E326 adds smoothed component-owner log odds derived only from lexemes in the
E316 training corpus. Record-held five-fold diagnostics reach 0.85–0.90
accuracy and 0.71–0.86 macro recall; the learned E318/E325 heads reach only
0.55–0.59 accuracy and about 0.10 macro recall. No evaluation aliases or rows
enter the prior.

The matched 446-step / 20,044-token CPU run took 137.67s. Checkpoint SHA:
`eb5683cf13231cae0f25b07fd66187c7e4534cb415e709093adeeb5109f363a8`.
Weighted/broad NLL are 5.4084/5.4961; loss AgentV passes 1/1. Final-20
row-weighted slot accuracy is 0.8026 versus batch-majority 0.6184.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.5458 | 0.6000 | 0.4000 | 0.5862 | Pass |
| adversarial | 4 | 1.0 | 1.0 | 0.6304 | 0.7500 | 0.6250 | 0.7238 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.5229 | 1.0000 | 0.5417 | 0.9857 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.4826 | 1.0000 | 0.5556 | 1.0000 | Pass |

Against E316, held-out meaningful/recall rise 0.40/0.20→0.60/0.40 and
adversarial meaningful/recall rise 0.50/0.375→0.75/0.625 without losing OOD
or limited-RICO gates. AgentV improves 3/5→4/5.

**Verdict:** retain E326 as the strongest scratch checkpoint. Do not promote
or claim ship: smoke component recall remains 0.3333 against a 0.35 gate.
