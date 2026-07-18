# E419–E420 three-step continuation boundary — 2026-07-18

E419 continues E396 for three deterministic power-zero optimizer steps,
ending at step 430 / 22,197 target tokens in 9.4 seconds. Checkpoint SHA is
`0b3c3bc7d04d5ffdbb477b0f8e2c36882399e7e1bd8bfbf8a3138c47cb542093`.
It is local-only, inherits best weighted NLL 5.8091 without a fresh loss
evaluation, and is not promoted.

E420 reproduces E418 exactly on the aggregate suites. Smoke meaningful rate is
0.6667, but type recall 0.3333 misses its 0.35 gate. Held recall is 0.5833;
AgentV is 3/4. Since one step passes, only the two-step checkpoint remains
unmeasured.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.6667 | 1.0 | 0.4628 | 0.3333 | 0.6607 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.6633 | 0.5833 | 0.7814 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5352 | 0.6042 | 0.7335 |

Every command used the external 290-second interrupt / ten-second forced kill;
training also used the internal 4.5-minute limit. E419 stopped normally on
token budget. E420 completed in 23.6 seconds and returned expected gate exit 8.
No timed-out process contributes evidence.

**Verdict:** reject E419. The exact bounded gate transition is step 429 or 430;
test the two-step checkpoint and retain E396 meanwhile.
