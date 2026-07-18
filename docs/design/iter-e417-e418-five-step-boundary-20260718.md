# E417–E418 five-step continuation boundary — 2026-07-18

E417 continues E396 for five deterministic power-zero optimizer steps, ending
at step 432 / 22,277 target tokens in 10.1 seconds. Checkpoint SHA is
`e74f46f0e45414ca3f30e89037a1676ecbce18346e898620ca01e859d6c6d177`.
It is local-only, inherits best weighted NLL 5.8091 without a fresh loss
evaluation, and is not promoted.

E418 narrowly fails the complete bounded gates. Smoke meaningful rate is
exactly 0.6667, but type recall 0.3333 misses the 0.35 floor by 0.0167.
Held recall improves to 0.5833 and structure to 0.6633; AgentV is 3/4.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.6667 | 1.0 | 0.4628 | 0.3333 | 0.6607 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.6633 | 0.5833 | 0.7814 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5352 | 0.6042 | 0.7335 |

Every command used the external 290-second interrupt / ten-second forced kill;
training also used the internal 4.5-minute limit. E417 stopped normally on
token budget. E418 completed in 22.8 seconds and returned expected gate exit 8.
No timed-out process contributes evidence.

**Verdict:** reject E417 on the authoritative recall floor. The bounded gate
boundary lies between 2 and 4 additional optimizer steps; retain E396.
