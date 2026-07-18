# E423–E427 low-learning-rate continuation — 2026-07-18

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
full RICO and checkpoint sync are absent.

## E426 fresh loss suites

E426 evaluates the frozen loss-suite v1 objective at mask rates
0.15/0.30/0.50/0.70/0.85, seed 0, over all five held-out and four OOD records.
The report is complete with no missing categories: weighted NLL is 5.2778,
improving on the inherited 5.8091. Category NLLs are binding 6.1367,
structural 3.1708, repair 7.8092, broad 4.6817, and schema OOD 4.0936. Its
finite diagnostic AgentV case passes 1/1. This is fresh selection evidence,
not a ship evaluation; E423 still lacks full RICO and remote checkpoint sync.

## E427 matched RICO slice

E427 evaluates the exact RICO rows 336–384 used by E396/E399. It completes all
48 rows with parse and fidelity 1.0, but meaningful rate is 0.9583, structure
0.6170, type recall 0.7326, and reward 0.9571. One row is trivial and one has
low component recall. E396 on the identical slice achieved meaningful 1.0,
structure 0.6401, recall 0.8993, and reward 0.9991. The diagnostic AgentV
envelope is 0/5 because four required suites are absent and RICO is a 48/1500
subset; there are no execution errors.

E427 completed normally under the external 290-second cap. A preceding command
pointed at the three-row fixture RICO directory and failed immediately on the
offset check; it ran no examples and contributes no evidence.

**Final verdict:** reject E423 as a replacement for E396. Lower LR repairs the
small-suite decode boundary and improves NLL, but materially regresses matched
RICO structure and type recall. Do not spend full-RICO coverage, sync, or
promotion resources on this checkpoint. Retain E396 while seeking a lever that
preserves both bounded and RICO quality.

## E428 component-plan decode-weight diagnostic

E428 repeats E427 with only component-plan decode weight increased from 2 to
4. Meaningful rate and type recall remain 0.9583/0.7326, while structure
regresses further from 0.6170 to 0.4947 and reward is 0.9568. The same one
trivial and one low-recall failure remain. AgentV is again an expected 0/5
diagnostic envelope with no execution errors. Increasing plan-head authority
amplifies the wrong hierarchy signal and is rejected.

E429 sets component-plan decode weight to zero. Structure improves to 0.6703,
above E396's 0.6401, and meaningful rate reaches 0.9792 with no trivial-layout
failure. Type recall, however, collapses to 0.5486 and one low-recall failure
remains. The plan head supplies needed component types but trades away
hierarchy quality; neither endpoint is acceptable as an E396 replacement.

E430 tests the sole interpolation point, component-plan weight 1. It remains
in the rejected plan-active regime: meaningful 0.9583, structure 0.6186,
recall 0.7326, and reward 0.9571, with the same two failures. Weight 1 does
not interpolate recall between weights 0 and 2; the response is discrete.
Close the scalar sweep. A separate hierarchy objective is required.
