# E431–E433 auxiliary-only continuation — 2026-07-18

E431 tests whether the component-plan and slot-component heads can continue
without moving the core denoiser. The new opt-in `--auxiliary-only` mode freezes
every parameter except active auxiliary heads after full-state restore and
fails clearly when the chosen objective has no differentiable auxiliary loss.
A regression test verifies that active auxiliary tensors change while every
base tensor remains byte-identical.

E431 resumes E396 on the unchanged 998-record E357 corpus at `lr=3e-4`. It
stops normally on the 23,000 target-token budget at step 446 / 23,019 tokens
after 20.6 seconds. The saved optimizer uses `lr=3e-4`; 183,309 parameters are
trainable and 135,765,696 are frozen. Independent checkpoint comparison finds
four changed tensors (`component_plan_head.{weight,bias}` and
`slot_component_head.{weight,bias}`) and zero changed base tensors. Checkpoint
SHA is
`be66555fe473b50eb17a76654ca9f12ddf5ae433f5d743d60eda3ff47c541ab3`.
The checkpoint is local-only, inherits best weighted NLL 5.8091, and is not
promoted.

E432 uses the matched 320-token honest LTR protocol:

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6304 | 0.6250 | 0.7238 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9828 |

AgentV passes 4/4 with no execution errors. Exit 8 reflects only the absent
full RICO suite. Compared with E396, the bounded screen improves OOD but
regresses held-out meaningful rate, structure, and recall.

E433 evaluates the exact RICO rows 336–384 used by E396/E399/E427. It completes
all 48 rows with parse and fidelity 1.0, meaningful rate 0.9792, structure
0.6211, type recall 0.6007, and reward 0.9768. One row has low component
recall. E396 on the identical rows achieved meaningful 1.0, structure 0.6401,
recall 0.8993, and reward 0.9991. The diagnostic AgentV envelope is 0/5 because
four required suites are absent and RICO is only 48/1500; it has no execution
errors.

Every command used an external 290-second interrupt and ten-second forced
kill; training also used the internal 4.5-minute wall limit. E431–E433 all
completed normally. Three path/schema probes failed immediately before
evaluating any examples and contribute no evidence. No timed-out process
contributes evidence.

**Verdict:** reject E431 as an E396 replacement. Auxiliary-only continuation
proves that base weights can remain exactly frozen, but the updated plan/slot
heads materially regress matched RICO type recall and structure. Do not run
full RICO, sync, promote, or make a ship claim from this checkpoint.
