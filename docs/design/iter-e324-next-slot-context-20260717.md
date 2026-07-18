# E324 ordered next-slot context — 2026-07-17

E324 encodes each current visible slot followed by the next slot, using only
contract text and preserving order. E318's diffusion recipe, data, seed,
architecture, and ordinary cross-entropy are fixed; the context flag is the
only behavior delta.

The 446-step / 20,044-token CPU run took 119.35s. Checkpoint SHA:
`53fcf5eec79d3a839d94f2474b4611439c94c638b2f95fdcf96a507db4991cb5`.
Weighted/broad NLL are 5.4238/5.5017; loss AgentV passes 1/1. Final-20 slot
accuracy is 0.6758 versus majority baseline 0.6392.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail recall |
| held_out | 5 | 1.0 | 1.0 | 0.4155 | 0.2000 | 0.1000 | 0.1994 | Fail meaningful/recall |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.3794 | 1.0000 | 0.5556 | 1.0000 | Pass |

AgentV remains 3/5 with three metric failures. Ordered concatenation changes
legal choices but gives no gate gain.

**Verdict:** reject the checkpoint and do not promote or claim ship. A single
pooled sequence does not learn the needed slot interaction; test separate
current/neighbor representations with an explicit interaction.
