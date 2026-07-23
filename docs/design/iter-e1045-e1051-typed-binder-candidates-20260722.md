# E1045-E1051 — typed binder-component candidates

Date: 2026-07-22. CPU scratch work under the repository wall cap.

v270 trains each detached binder-component row against only the component
classes permitted by that binder's schema-typed use sites. It derives those
classes from grammar state and the official component schema; it does not use
natural-language target literals or placeholder text. The active E937/E938
boundary contains 1,214 audited primary and alternate targets with zero
role-contract violations.

E1045 starts fresh with no parent and trains the same joint binder-component
and binder-arity weights as E1029. Its internal 95-second wall budget stops it
cleanly at 395 of 450 requested steps after 1,580 examples. This is valid
bounded diagnostic evidence, but not a step-matched replacement for E1029.
Checkpoint SHA is
`dc39d2f12391af21be7aeeefad42fca3ff45b01f32dfc2b43dcdb6481a22eef1`;
sync is explicitly disabled.

Typed candidate count falls from the former fixed 35 classes to 15.46 on the
final batch. Final binder-component loss/accuracy are 0.9043/0.7692; their
last-50 means are 1.8523/0.4304. Final binder-arity loss/accuracy are
1.1607/0.5.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1046 | smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5658 | 0.75 | 0.9610 | 0 / 0 |
| E1047-E1051 | five held one-row subsets | 5 | 0.8 | 0.6 | 0.6667 | 0.3762 | 0.5333 | 0.7132 | 1 / 3 |
| E1034-E1038 v268 control | five held one-row subsets | 5 | 0.6 | 0.4 | 0.44 | 0.2395 | 0.4 | 0.5106 | 2 / 3 |
| E996 retained baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

E1047-E1051 are completed one-row held diagnostics under one identical policy;
their arithmetic means are not a canonical full-suite scoreboard or ship
evaluation. Dual Card becomes strict-valid with full component recall, and
Input stays strict-valid. Form still times out, Tabs retains an empty
`Tabs([])`, and Settings collapses to one `TextContent`. All six evals emit
AgentEvals JSONL and pinned AgentV bundles (`0/6`).

Retain the v270 typed-supervision capability, but reject E1045: it remains
below the E996 retained baseline on every held headline metric. Never sync,
promote, serve, resume, or use it as a parent. The next experiment should
isolate the remaining untyped rows or binder-arity interaction rather than
return to open-string supervision.

## E1052-E1057 arity isolation

E1052 disables only binder-arity decode. Smoke remains strict-v2 1.0 with
recall 0.75 and structure rises slightly to 0.5717. Five held one-row
diagnostics then reach parse/strict/fidelity/structure/recall/reward
0.6/0.6/0.6/0.3562/0.5333/0.5646 with two timeouts and two fallbacks. Settings
improves from a one-`TextContent` collapse to a strict-valid Slider/Switch
layout, but Dual Card changes from strict-valid to a timeout. All six runs emit
AgentEvals JSONL and pinned AgentV bundles (`0/6`).

Reject the no-arity decode policy. Arity helps declaration reachability even
while harming particular rows, so the next arm needs calibrated or
typed-conditional arity rather than a global on/off switch.

## E1058-E1062 arity calibration

E1058 tests binder-arity weight 0.5. Smoke is prediction-identical to arity-off:
the head applies 15 times but changes no choices. E1059-E1060 probe only the
two held rows that flip under arity on/off. Dual Card still times out, while
Settings remains strict-valid with structure 0.41 and full component recall.

E1061-E1062 repeat those rows at weight 0.75. The policy crosses both decision
thresholds: Dual Card reproduces the strict-valid weight-1 result, while
Settings reproduces the one-`TextContent` collapse. All five runs emit
AgentEvals JSONL and pinned AgentV bundles (`0/5`).

Close the scalar sweep. No global arity weight serves both rows. The grammar
already has a dedicated root-reference arity head, so the next minimal
representation arm will isolate binder arity to bound declarations instead of
letting it compete for root-list ownership.

## E1063-E1065 bound-declaration ownership

v271 excludes the root declaration from the generic binder-arity training and
decode rows; the dedicated root-reference arity head remains the sole owner of
root-list cardinality. E1063 confirms that the generic arity head abstains on
all smoke choices: smoke is prediction-identical to the arity-off arm at
strict-v2 1.0, structure 0.5717, recall 0.75, and reward 0.957.

The two opposed held rows remain opposed. E1064 Dual Card times out to an empty
prediction, while E1065 Settings is strict-valid with fidelity 1.0, structure
0.60, recall 1.0, and reward 0.937. Each run emits AgentEvals JSONL and a pinned
AgentV bundle (`0/3`).

Retain v271 as an ownership correction, but reject this decode policy and keep
E1045 non-parentable. The Dual Card gain at higher global arity weights came
from root-list pressure, not reusable bound-declaration evidence. A subsequent
arm must model root-list identity/cardinality through its dedicated owner
rather than restoring overlapping generic supervision.

## E1066 dedicated root-reference arity train

E1066 is a fresh CPU scratch train on the same audited 524-row E937 corpus. It
combines typed binder-component loss 1, bound-only generic binder-arity loss 1,
and dedicated root-reference arity loss 1 under compiler-tree capability. The
95-second internal wall budget stops cleanly at 331/450 requested steps after
1,324 examples in 95.24 seconds. It has no parent and checkpoint sync is
explicitly disabled.

The final root-reference arity loss/accuracy are 1.2019/0.50 over four rows;
their last-50 means are 1.3583/0.5367 over 160 rows. Final binder arity
loss/accuracy are 0.6901/0.80, and final binder-component
loss/accuracy/candidate count are 2.0869/0.2857/28.43. Checkpoint SHA is
`51ffed4a5ede84a8422a5402b14688adda3bc51d83bc9c58ebc270d281b8ff22`.

This is a completed bounded diagnostic checkpoint, not a production or
step-matched result.

E1067 evaluates strict smoke with all three matching decode heads at weight 1.
Parse and meaningful-v1 remain 1.0, but strict-v2/fidelity/structure/recall/
reward are 0.6667/0.9167/0.6675/0.6667/0.9320. The Hero row omits a required
placeholder. There are no timeouts or fallbacks, and the run emits AgentEvals
JSONL plus a pinned AgentV bundle (`0/1`).

Reject E1066 without held evaluation. Its structure gain over E1063 does not
offset the smoke contract and component-recall regressions. Never sync,
promote, serve, resume, or use it as a parent.

E1068 disables only root-reference arity decode on the same checkpoint.
Fidelity returns to 1.0 and reward to 0.957, but strict-v2 and component recall
remain 0.6667; Hero now duplicates a placeholder identity instead of omitting
one. This separates a single harmful root-head decode choice from the broader
checkpoint regression. The root head consumes detached context, so these runs
do not establish shared-gradient interference. The remaining causal
candidates are the shorter 331-step exposure and the changed RNG trajectory
from instantiating an additional head, not active data content. AgentV is
`0/1`.

## E1069 matched-exposure control

E1069 removes only the dedicated root-reference objective while preserving the
E1066 seed, E937 data, compiler-tree mode, batch size, typed binder-component
loss, and bound-only binder-arity loss. The first bounded invocation reaches
316/331 requested steps and stops cleanly at 95.32 seconds after 1,264
examples. It is therefore not yet the intended 331-step control and no causal
comparison is made. The local checkpoint SHA is
`9f4a52e62a7797ec81cdc2b02fbd915ea1bf97c2de9207a074d0531d9213bcb0`;
sync is disabled. Resume only its own full-state checkpoint for the remaining
15 steps before evaluation.
