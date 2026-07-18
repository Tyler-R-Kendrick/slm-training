# E415–E416 ten-step continuation boundary — 2026-07-18

E415 bisects the safe one-step E413 and failed 19-step E411 continuations. It
resumes the same E396 full state with balance power zero and stops after 10
additional optimizer steps: step 437 / 22,561 target tokens in 14.2 seconds.
Checkpoint SHA is
`99e3b6029e352e1b4066175733b3bb24e3778227d7879c14f01376d75b0d1fa5`.
It is local-only, inherits best weighted NLL 5.8091 without a fresh loss
evaluation, and is not promoted.

E416 fails smoke meaningful rate and type recall at 0.3333 / 0.1667. Held
recall improves to 0.5833, but that does not compensate for the smoke failure.
AgentV is 3/4.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.4975 | 0.1667 | 0.3203 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5924 | 0.5833 | 0.7790 |
| adversarial | 4 | 1.0 | 0.5 | 1.0 | 0.5137 | 0.3750 | 0.4865 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5352 | 0.6042 | 0.7335 |

Every command used the external 290-second interrupt / ten-second forced kill;
training also used the internal 4.5-minute limit. E415 stopped normally on
token budget. E416 completed in 23.1 seconds and returned expected gate exit 8.
No timed-out process contributes evidence.

**Verdict:** reject E415. The smoke regression begins between 2 and 9
additional optimizer steps; retain E396 and continue the bounded bisection.
