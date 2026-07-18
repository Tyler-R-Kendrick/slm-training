# E325 explicit slot-pair interaction — 2026-07-17

E325 separately encodes the current and next visible slots and adds their
elementwise interaction to the slot-owner head. E318's data, seed,
architecture, objective, and decode weight are fixed; a missing neighbor
exactly reduces to the current-slot baseline.

The 446-step / 20,044-token CPU run took 123.64s. Checkpoint SHA:
`e0f8e1266ba3199a4ee2dfda19e46a9e617541701ddeb96e27dc6d1ee7c8da6b`.
Weighted/broad NLL are 5.4328/5.5050; loss AgentV passes 1/1. Final-20
row-weighted slot accuracy is 0.7105 versus batch-majority baseline 0.6184.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4431 | 0.4000 | 0.2000 | 0.3916 | Fail recall |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.3850 | 1.0000 | 0.5556 | 1.0000 | Pass |

AgentV is 3/5 with the same two gate failures as E316, but this is not a
quality tie. The interaction changes 16 of 19 predictions and cuts E316 OOD
meaningful/recall/reward from 1.00/0.5417/0.9857 to
0.50/0.25/0.4992. It recovers held-out quality lost by E322–E324 without
clearing either remaining gate.

**Verdict:** reject the checkpoint and do not promote or claim ship. E316
remains the strongest scratch checkpoint. Adjacent position is too weak a
relation; the next lever should use semantic role features derived from the
slot names rather than another loss reweighting or positional neighbor.
