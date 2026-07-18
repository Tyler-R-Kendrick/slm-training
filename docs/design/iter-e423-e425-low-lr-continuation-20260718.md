# E423–E425 low-learning-rate continuation — 2026-07-18

Full-state resume previously restored the checkpoint optimizer parameter-group
learning rates after constructing AdamW with the requested CLI rate. The train
summary nevertheless reported the requested rate. The harness now reapplies
`config.lr` after optimizer-state restoration; a regression test verifies both
the saved optimizer groups and summary recipe, while the existing same-rate
resume remains bit-exact.

E423 resumes E396 on the unchanged 998-record E357 corpus and changes only the
learning rate from `3e-4` to `3e-5`. It executes the same 19 batches as E411,
ending at step 446 / 23,019 target tokens in 21.6 seconds. All five saved AdamW
parameter groups contain `lr=3e-5`. Checkpoint SHA is
`2a6b84ba7259937bbaf1e3edb712f2adde4ce8cdef0be6cbcc55e7ffa260e3ad`.
The run is local-only, inherits best weighted NLL 5.8091 without a fresh loss
evaluation, and is not promoted.

E424 is a complete four-suite bounded evaluation with the CLI's 256-token LTR
default. It passes AgentV 4/4, but is a protocol variant because E412 used an
explicit 320-token LTR budget. E425 repeats the evaluation with the matched
320-token policy and produces identical aggregate quality:

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.6633 | 0.5833 | 0.7814 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 0.75 | 1.0 | 0.5352 | 0.6042 | 0.7335 |

At the same step and token boundary, high-rate E411/E412 had smoke
meaningful/recall 0.3333/0.1667 and AgentV 3/4. E423/E425 instead preserve
the safe Button prediction and improve held structure/recall from
0.5524/0.4833 to 0.6633/0.5833. This supports optimizer sensitivity, rather
than malformed step-430 data, as the immediate cause of the discrete
Button-to-TextContent flip.

Every command used an external 290-second interrupt and ten-second forced
kill; training also used the internal 4.5-minute limit. E423 stopped normally
on token budget. E424 and E425 completed normally and returned exit 8 only
because full RICO is absent. One earlier E424 invocation failed immediately
on a missing default test directory and contributes no evidence. No timed-out
process contributes evidence.

**Verdict:** retain E423 as the stronger bounded continuation candidate and
use `3e-5` for further controlled continuation. Do not promote or claim ship:
fresh loss evidence, full RICO, and checkpoint sync are absent.
