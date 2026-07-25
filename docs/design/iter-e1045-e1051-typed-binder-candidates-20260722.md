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

The own-checkpoint resume completes the remaining 15 steps in 5.60 seconds,
ending exactly at 331 with cumulative train wall 100.92 seconds and 1,324
examples. Final loss is 6.2251; binder-component
loss/accuracy/candidates are 2.0787/0.2857/28.43 and binder-arity
loss/accuracy are 0.6897/0.80. The completed control checkpoint SHA is
`79d3f93dfbf5a5fbe0388931aef3ec9c07d69adab9fe73f6f4248a578b8f5617`.
It is now pending smoke evaluation.

E1070 applies the same root-decode-off smoke policy as E1068. All three
predictions and all headline metrics are identical: parse/meaning-v1 1.0,
strict-v2 0.6667, fidelity 1.0, structure 0.6633, recall 0.6667, and reward
0.957, with Hero duplicating one placeholder identity. AgentV is `0/1`.

This exact behavioral control closes the attribution: the detached root
objective does not cause the base-model regression. The 331-step exposure is
insufficient, while enabling the trained root head adds the separate harmful
choice observed in E1067. Reject E1069 at 331 steps and do not promote or use it
as a parent.

## E1071 v271 395-step exposure

E1071 starts fresh on the same audited E937 data with v271 bound-only generic
arity and typed binder-component loss, both at weight 1. Its first bounded
invocation reaches 342/395 steps after 1,368 examples in 95.18 seconds and
stops cleanly on the internal wall budget. Final binder-component
loss/accuracy/candidates are 1.5942/0.4444/30.11; binder-arity loss/accuracy
are 0.5220/0.8182. The partial checkpoint SHA is
`0dbeed6453e64e3a28ce38661f0f80b2c2851fd468cc6b529313186136935832`.
It is not evidence for the 395-step hypothesis; resume only its own full-state
checkpoint for the remaining 53 steps before evaluation.

The own-checkpoint continuation completes those 53 steps in 11.96 seconds.
E1071 ends exactly at 395 steps after 1,580 examples and 107.13 cumulative
train seconds. Final binder-component loss/accuracy/candidates are
0.9040/0.7692/15.46, closely reproducing E1045; bound-only binder-arity
loss/accuracy are 1.0203/0.6154. The completed checkpoint SHA is
`0d2c4c4806d7815906a110aff4ec0d9029823d202447ba87de40b641dc6f9a12`.
It is pending strict smoke and remains local, unsynced, and non-parentable.

E1072 strict smoke reaches parse/meaning-v1/strict-v2/fidelity 1.0, structure
0.5717, recall 0.75, and reward 0.957 with no timeout or fallback. All three
predictions are identical to E1063 and the arity-off E1052 arm. The run emits
AgentEvals JSONL plus a pinned AgentV bundle (`0/1`). Proceed only to the
targeted Dual Card and Settings rows; the checkpoint remains pending held
disposition.

E1073 Dual Card times out to an empty prediction. E1074 Settings is
strict-valid with fidelity 1.0, structure 0.60, recall 1.0, and reward 0.937.
These exactly reproduce E1064-E1065: additional exposure restores clean smoke
but does not create a useful bound-only arity signal for the root-list
decision. All three E1072-E1074 evaluations emit AgentEvals JSONL and pinned
AgentV bundles (`0/3`).

Reject E1071 without a redundant full-held run. Never sync, promote, serve,
resume, or use it as a parent. Retain v271's ownership correction because it
prevents the Settings collapse, but the next quality arm must improve root-list
construction through a different representation rather than generic binder
arity or another scalar/exposure sweep.

## E1075 v272 lexer-native root identity

v272 generalizes the detached root-reference identity target and learned
re-ranking path from choice sections to lexer-native binder slots. It normalizes
`bind1` to identity 0 and never derives targets from prompt or string surface
text. The active E937 loader remains fail-closed for every primary and alternate
target; its 524 train rows plus all 50 E938 eval rows pass the durable role-safe
loader test. The lexer-native strict-subset audit finds 220/524 train records
whose root uses a nonempty proper subset of declared binders, so this objective
has observed positive and negative supervision.

E1075 starts fresh with the E1071 recipe and 395 requested steps: CPU scratch,
batch 4, audited E937 data, compiler-tree decode, typed binder-component loss
1, bound-only binder-arity loss 1, and the new root-identity loss 1. The first
bounded invocation stops cleanly on its internal wall budget at 378 steps /
1,512 examples in 95.13 seconds. Final root-identity
loss/exact/positive-recall/negative-accuracy/classes are
0.6189/0.3333/1.0/0.0/3.67; binder-component
loss/accuracy/candidates are 1.8367/0.3333/35.0, and binder-arity
loss/accuracy are 0.2137/0.9091. The partial serving checkpoint SHA is
`ffd06d4fcc7e76ab7ea80c7f801266e114b5e3188d3c2a9e55284f61d3792a2e`.

This partial is not quality evidence and must not be evaluated, synced,
promoted, served, or used as a parent. Resume only its own serialized full state
for the remaining 17 steps, then document the completed checkpoint before
strict smoke.

The own-state continuation completes the remaining 17 steps in 5.03 seconds.
E1075 ends exactly at 395 steps / 1,580 examples and 100.16 cumulative train
seconds. Final root-identity
loss/exact/positive-recall/negative-accuracy/classes are
0.5937/0.6667/1.0/0.25/4.33. Binder-component
loss/accuracy/candidates are 0.9040/0.7692/15.46 and binder-arity
loss/accuracy are 1.0203/0.6154, exactly matching E1071 at step 395; the
detached isolated identity objective did not perturb those base objectives.
The completed checkpoint SHA is
`9e03575396dba1aa971e2aa29224023ed9bfbc053a83f48414ef14eb872122c7`.
It remains local, unsynced, non-parentable, and pending strict smoke with
root-identity decode enabled.

E1076 strict smoke reaches parse/meaning-v1/strict-v2/fidelity 1.0, structure
0.5396, recall 0.75, and reward 0.965 with no timeout or fallback. The
root-identity path applies 15 times and changes two choices: Hero and Button
use a distinct `b2` reference where E1072 reused `b1`. All three predictions
pass the role-safe output assertion. Relative to E1072, strict/fidelity/recall
are unchanged, reward rises from 0.957, and structure falls from 0.5717, so
this is a mixed but reachable intervention rather than a smoke-quality
promotion. The named evaluation criteria do not pass. Proceed only to the
targeted Dual Card and Settings rows before disposition.

E1077 evaluates the targeted Dual Card row. It still reaches the canonical
12-second decode timeout and produces an empty prediction, exactly matching
E1073 at parse/strict 0.0. The named evaluation criteria fail; root identity
does not repair the failure-defining row. Run only the matched Settings
diagnostic before rejecting or retaining the checkpoint.

## E1079 v272 root-identity reproduction (invalid transport interruption)

E1079 begins a fresh, local-only reproduction of the E1075 recipe to recover a
valid decode-off control without resuming or parenting the rejected E1075
checkpoint: audited E937, CPU scratch, lexer/tree compiler path, batch 4, seed
0, 395 requested steps, typed binder-component loss 1, bound-only binder-arity
loss 1, and root-identity loss 1. The CLI capability preflight first rejected
the default non-lexer configuration; the corrected lexer/tree command then
started successfully.

The local command transport terminated the corrected process at approximately
30 seconds, before the harness's derived wall budget and before it serialized a
checkpoint or `train_summary.json`. Its last metrics line is step 71 with loss
13.6302 and root-identity loss/exact/positive-recall/negative-accuracy
0.7050/0.0000/0.8333/0.0000. This is an invalid interrupted run, not training
or quality evidence: it must not be resumed, evaluated, synced, promoted,
served, or used as a parent. Restart a fresh E1079-equivalent arm through a
persistent terminal session, then evaluate only a fully serialized checkpoint.

## E1080 v272 root-identity reproduction (own-state partial)

E1080 restarts E1079's E1075-recipe reproduction in a persistent terminal with
the same E937 manifest, CPU scratch lexer/tree path, batch 4, seed 0, and the
three structural loss weights of 1. Its first bounded call cleanly reaches
37/395 steps in 96.03 seconds and serializes its own full-state checkpoint. The
own-state continuation then reaches exactly 395 steps in 83.90 seconds, for
179.94 cumulative train seconds. The completed local-only checkpoint SHA is
`eae1afd2bdfe587538e0bd2a44edc6a87600fb897f02970a10d3a723bc54623d`; the
strict-subset audit remains 220 rows and final loss is 5.7942.

The first invocation's throughput differs materially from E1075's host, so
loss and wall time are not quality comparisons. E1080 is a completed scratch
diagnostic only: it remains unsynced, unpromoted, unserved, and non-parentable.
It is now eligible solely for the preregistered decode-off Settings diagnostic;
the named AgentEvals grader result determines whether the earlier Settings
regression was decode-ranking-specific or checkpoint-level.

E1081 runs that Settings row with the identity decode weight explicitly zero.
It completes without a timeout (parse 1.0, timeout 0), unlike E1078's empty
12-second result, but it is not meaningful: strict-v2 0, fidelity 0.3333,
structure 0.0600, component recall 0, and reward 0.707. These named domain
metrics are evaluated through the AgentV SDK; AgentV itself has no score.
This is neither a ship result nor a quality gain. Because E1081 uses a fresh
checkpoint produced under a later dirty code state, it cannot by itself assign
the E1078 timeout to ranking. Run the paired identity-decode-on Settings replay
against the exact same E1080 checkpoint next.

E1082 is that paired replay, differing only by identity decode weight 1. The
path is active (17 applications and seven changed choices), yet every reported
final metric is identical to E1081: parse 1.0, strict-v2 0, fidelity 0.3333,
structure 0.0600, component recall 0, reward 0.707, and no timeout. The eval
component stamps match; concurrent work changed the code
commit between E1081 and E1082, which is disclosed in both stamped payloads.
Identity ranking is therefore not a sufficient explanation for E1078's timeout.
Reject E1080 as a scratch checkpoint: both paired Settings paths are
strict-invalid. The next hypothesis must change root-list representation, not
continue scalar ranking sweeps.

## E1278-E1285 required-component ProgramSpec corpus loop

This bounded data-only loop tests whether ProgramSpec generation can create
strict, natural-language examples with TextContent, Form, and Input in the
same topology. No model was trained, evaluated, promoted, or synced from these
snapshots.

| Build | Recipe | Admitted | Result |
| --- | --- | ---: | --- |
| E1278 broad | ordered trio with default derivative producers | 128 / 962 | Non-promotable: 358 normalization errors, 102 verifier quarantines, 60 quality rejections, and multiple feedback warnings make it unsuitable as a topology isolate. |
| E1278 direct | same roots; only direct ProgramSpec records | 3 / 20 | Non-promotable: 9 verifier quarantines, 4 quality rejections, and 3 decontamination drops. |
| E1280 direct | hard required trio with natural prompts | 0 / 20 | Rejected producer diagnosis: Form implicitly realizes Buttons, while the prompt named `buttons`; the independent component judge searched for singular Button. |
| E1281 direct | omit implicit Buttons/Stack from natural prompts | 8 / 20 | Strictly clean: parse and judge 1.0, zero quarantines, zero quality rejects, zero n-gram flags. Dedup removes 60%, so it is too small to train. |
| E1282 direct | E1281 producer at 100 roots | 9 / 100 | Strictly clean but saturated: 29 exact, 33 fuzzy, and 29 semantic duplicates. The feedback recommendation is producer diversity, not a gate change. |
| E1283 direct | optional Card superstructure | 8 / 20 | Strictly clean but no yield gain; Card survives in only one of eight records and dedup remains 60%. |
| E1284 direct | force Card with the required triple | 8 / 20 | Strictly clean and Card appears in all eight records, but 12 semantic duplicates remain. The current builder makes every component a sibling, so permutations are not topology diversity. |
| E1285 direct | schema-valid Card containment | 7 / 20 | Strictly clean but worse than E1284: 5 fuzzy and 6 semantic duplicates. Card containment alone does not create a usable standalone corpus. |

The durable source reports are the immutable snapshots under
`src/slm_training/resources/data/train/`; their embedded `version_stamp/v1`
records preserve the actual producer versions. The E1282 result falsifies
simple expansion as a remedy: increasing seed count from 20 to 100 adds one
admitted record while duplicate share rises from 0.60 to 0.91. The next arm
is closed as non-promotable data evidence. The next model run returns to the
524-row E937 role-safe corpus; no checkpoint will be trained on the 7-9 row
ProgramSpec isolates.

## E1286 fresh E937 control (invalid transport interruption)

E1286 starts a fresh CPU scratch control on immutable E937 with the E1080
training recipe and root-reference decode disabled. The command transport
ended at step 203 before the harness serialized a checkpoint or
`train_summary.json`. Its final streamed loss is 8.2223, but that partial
stream is not training or quality evidence. Do not resume, evaluate, sync,
promote, serve, or parent it; rerun the fresh arm only through a persistent
terminal session.

## E1287 fresh E937 own-state control (smoke only)

E1287 is the fresh replacement for invalid E1286: the immutable 524-row E937
role-safe corpus, CPU scratch lexer/tree path, batch 4, seed 0, 395 steps, and
the E1080 structural losses with root-reference identity decode explicitly
zero. The harness resumed its own serialized state across two wall-budget
boundaries and completed at step 395; it did not resume E1286. The local
checkpoint SHA is `52974b70a6119ea6baa360e4c58bbba40840e10c67a2c3f31f8f39fc7ed1dc3e`.

Strict smoke n=3 has parse, meaningful-program, strict-v2, and placeholder
fidelity all 1.0; structural similarity/tree edit similarity 0.5717; component
recall 0.75; reward 0.957; and no decode timeouts. The `@agentv/core` SDK
executed all 25 named domain graders with zero execution errors; no SDK
aggregate is reported or used as a model metric. This is a single-suite local
scratch diagnostic, not a ship result and not eligible for sync, promotion,
serving, or parent use. Its next bounded evidence is the same-checkpoint
held-out Settings diagnostic, not a new training variant.

## E1288 E1287 Settings decode-off diagnostic

E1288 evaluates only held_out_settings_01 (n=1, offset 4) from E938 against
the E1287 checkpoint under the same strict compiler-tree policy with
root-reference identity decode weight zero. It completes in 5.01 seconds with
no timeout, but is strict-invalid: parse 1.0, meaningful-program/strict-v2 0,
placeholder fidelity 0.3333, structural similarity 0.0600, component recall
0, and reward 0.707. The @agentv/core SDK executed all 25 named domain graders
with zero execution errors; those named values, rather than an SDK aggregate,
are the reported metrics. This eliminates the smoke result as a readiness
signal and keeps the checkpoint non-promotable. E1289 will run the
preregistered same-checkpoint decode-on pair.

## E1289 E1287 Settings decode-on pair

E1289 differs from E1288 only by root-reference identity decode weight 1. The
same checkpoint and held-out Settings record now produce an empty prediction
at the 12-second decode limit: parse, meaningful-program, strict-v2,
placeholder fidelity, structural similarity, component recall, and reward are
all 0; the decode timeout count is 1. The @agentv/core SDK again executed all
25 named domain graders with zero execution errors. Root-identity ranking
therefore worsens this same-checkpoint target rather than repairing it.

Reject E1287 and close this scalar decode sweep. Its smoke result is not
representative, decode-off is structurally invalid, and decode-on regresses to
timeout. The next hypothesis must alter the root-list representation or its
deterministic completion constraints, not train another root-identity scalar
variant.

## E1290 timeout-trace replay

E1290 replays E1289 exactly after evaluator v58 preserves the active
`DecodeStats` object when the signal timeout interrupts batch generation. The
same named `@agentv/core` graders again report parse/meaning/strict-v2 0 and
one 12-second timeout with zero SDK execution errors, so this is
instrumentation-only evidence, not a new quality result. The retained trace
shows 19 emitted tokens and two root-identity applications with zero choice
changes: the decoder first selects one forward binder reference, then chooses
an inline Slider continuation and exhausts the decode budget. The frontier is
therefore legal but incorrectly ranked; a scalar root-identity reweight is not
the next intervention.

## E1291-E1292 document-generation control

E1291 corrects the stronger training/evaluation mismatch exposed by the E1290
trace. E1287 trained on the E937 all-target corpus, whose admitted records also
include lexical and fragment scope slices, while the target evaluator requests
complete documents. The canonical strict builder now retains only
`target_kind=document`: 350 of 1,077 candidates are admitted, all 350 are
complete `root = ...` documents, parse and independent-judge rates are 1.0,
and 637 excluded scope targets remain explicitly recorded in `rejected.jsonl`.
It retains 82 Card and 7 Settings-shaped programs; no admission gate changed.

E1292 is a fresh local CPU scratch control on E1291, holding E1287's lexer/tree
architecture, seed 0, batch size 4, learning rate 0.0003, binder-component,
binder-arity, root-identity, and fidelity losses fixed. It completes three
data passes (263 steps) through four own-state wall-budget continuations; SHA
`7d48e2b91d6b14f1ffce523e6dcef221883b38d215af34fc5575bbf3fdf852b8` remains
local-only and non-promotable. The failure-defining Settings n=1 diagnostic no
longer times out: parse and v1 meaningfulness are 1.0, structural similarity
0.5200, component recall 0.5, and reward 0.749. The named strict-v2 grader is
still 0 because `:slot_1` and `:slot_2` are missing and `SwitchGroup.items` has
the wrong role; fidelity remains 0.3333. The `@agentv/core` SDK executed all 25
named domain graders with zero execution errors and no SDK aggregate metric.
The full held-out n=5 follow-up completes without timeout or fallback: parse
1.0, v1 meaningfulness 0.8, structural similarity/tree edit 0.6087, component
recall 0.6524, and reward 0.7632. Strict-v2 remains 0/5: every row misses a
required slot, with Form, Tabs, and Settings also placing content in an invalid
schema role. E1292 is rejected, unsynced, and never a promotion, serving, or
parent candidate. The next hypothesis is a deterministic request-contract
coverage constraint that permits only schema-compatible realization, not a
reopened slot-margin scalar sweep.

## E1293 required-slot root-completion decode control

E1293 evaluates the same E1292 Settings row with the new default-off
root-close constraint enabled. It avoids timeout but reaches one constrained
dead end; retry returns `root = TextContent(":slot_0")`. Parse remains 1.0,
but v1/strict-v2 are 0, fidelity 0.3333, structure 0.0600, recall 0, and
reward 0.707. The `@agentv/core` SDK executed all 25 named domain graders with
zero execution errors. Reject this root-only completion rule; it does not prove
a schema-compatible route to the remaining slots and must not be widened.

## E1294 fidelity-1 document control

E1294 is the only remaining matched scalar check justified by the E1292
full-document result: it repeats E1292's fresh CPU scratch recipe on E1291,
but raises `fidelity_loss_weight` from 0.5 to 1.0. The run completes all 263
steps with the same lexer/tree decoder, seed 0, batch size 4, learning rate
0.0003, binder-component, binder-arity, and root-identity losses. The final
local-only checkpoint SHA is
`a737aa49977ee046870cd7cd38a6007193a695c1d16ca06855d9d83f814cec1f`.

An initial invocation resolved the default `smoke` suite and selected zero
rows, so it is retained as an explicit non-evidence preflight. The corrected
matched held-out Settings n=1 probe (offset 4, strict compiler-tree policy,
root completion disabled) against the final SHA produces an empty 12-second
timeout: parse, v1 meaningfulness, strict-v2, fidelity, structure, tree edit,
component recall, and reward are all 0. The named strict grader reports parse
unavailable, missing required placeholders/components, unavailable binding
analysis, and an empty verifier input. The `@agentv/core` SDK executed and
published all 25 named domain graders with zero execution errors; it is not
reported as a metric. Reject E1294 without a wider suite. This closes the
fidelity scalar lever for this corpus: it regresses the E1292 Settings behavior
and the checkpoint must never be synced, promoted, served, resumed, or used as
a parent.

## E1295 rejected required-switch topology corpus

E1295 builds a strict document-only corpus with 96 ProgramSpecs that require
Slider, SwitchGroup, SwitchItem, and TextCallout together. The first producer
version only realizes structural `name` fields and empty SwitchGroup item
arrays, leaving zero content placeholders. The quality gate rejects these
records rather than accepting an opaque-marker violation: 189 of 2,039
collected documents remain, with 960 quality rejections. The immutable corpus
fingerprint is `350b8a6da2ce3523d586f2eb184c6fd1ed5f0253ef8662411209e94834d6cc33`.
It is not a training corpus. The repair is to use the typed schema's nonempty
SwitchGroup item reference and content-bearing Slider/TextCallout properties;
the quality threshold remains unchanged.

## E1296-E1297 strict switch-topology producer controls

E1296 first substitutes a schema-recognized TextContent carrier for the
unsupported composite child path. Its direct, no-augmentation ProgramSpec
build admits 33 of 100 candidates with zero quality rejections (55 semantic
deduplications and 12 verifier quarantines); every admitted row contains
Slider, SwitchGroup, TextCallout, and TextContent with visible slot vocabulary.
The immutable content fingerprint is
`8f5353866f3a3131e06d9260158ae9d9d0fa5d8b13199ca02dab2d0a24cfff43`.
This is clean producer evidence, but a 33-row isolate is too small to train.

E1297 restores SwitchItem and adds typed placeholder/nonempty-array candidates.
It admits 27 of 100 direct candidates, but the producer still mixes base
structural-only candidates: 47 are correctly rejected for too few
placeholders. Those same rows also expose a separate quality-vocabulary defect:
SwitchItem was preferred while its schema-supported SwitchGroup parent was not.
The vocabulary alignment is corrected in model/two-tower component v274 without
altering any threshold; it does not rescue the independent missing-placeholder
failure. E1297 is likewise not trainable. The next producer revision must emit
only placeholder-bearing candidates for an explicitly required topology.

## E1298 accepted multi-slot switch topology corpus

E1298 preserves strict gates and emits two SwitchItems inside each required
SwitchGroup, alongside a single legal TextCallout content field. It admits 442
documents: 270 contain Slider, SwitchGroup, SwitchItem, and TextCallout, and
194 of those have at least three content slots. The sampled four-slot composite
passes the parser and independent G11 judge with nested SwitchItems. Its
fingerprint is `cbc4c5c29395399d8facaf302a802252332c2a188abb7ebb0d140ead6d30d6cb`.
The corpus is eligible only for an E1301 local matched scratch control; it is
not a checkpoint promotion or ship result.

## E1300 invalid interrupted E1298 train attempt

E1300 began the intended fresh E1298 CPU control but was externally terminated
at step 112 before its first checkpoint or train summary. It has no resumable
state and is not evaluated, compared, synced, promoted, served, or used as a
parent. E1301 restarts the same recipe from a fresh seed-0 initialization under
the trainer's own wall-budget continuation mechanism.

## E1299 direct multi-slot switch producer control

E1299 isolates E1298's v37 producer without augmentation: 100 natural-prompt
ProgramSpecs requiring Slider, SwitchGroup, SwitchItem, and TextCallout. The
strict build retains 22 rows with mean quality 1.0 and zero quality or
decontamination rejects, but quarantines 19 at independent judge G11 and
deduplicates 59 further candidates (15 exact, 10 fuzzy, 34 semantic). Its
immutable content fingerprint is
`28e7128e13c41b865ca1f7cc232b9f7e92f224487e04699d70f42dcc7621af36`.
This proves the retained forms are schema-valid but falsifies a standalone
direct corpus as the E1298 training arm: 22 rows cannot replace E1291's
350-row document control. Preserve the rejection ledger and use E1298 only
for the already-scoped matched scratch mixture comparison; do not relax
deduplication, the independent judge, or any quality gate.

## E1301 multi-slot corpus train control

E1301 restarts the pre-scoped E1298 control and reaches the harness-derived
three-pass endpoint of 332 steps over 442 strict documents through its own
full-state continuations. The local-only checkpoint SHA is
`1980e8a3dc18f67b4f8aaafb6a6dfe151302903f5bbeab6962c918386955f390`; no
bucket sync occurred. The strict compiler-tree Settings n=1 diagnostic parses
without a timeout, but emits `root = TextContent(":slot_0")`: v1 and strict-v2
are 0, fidelity is 0.3333, structure is 0.06, component recall is 0, and reward
is 0.707. The learned decode selected `Stack`, `Slider`, `SwitchGroup`, and
`SwitchItem`, then hit `empty_completion_forest` inside the second SwitchItem;
one certified fallback returned the one-node result. Relative to E1292's
Settings row, structure regresses 0.52→0.06, recall 0.5→0, and reward
0.749→0.707. The pinned `@agentv/core` bundle executed and published 25 named
domain graders with zero execution errors; AgentV itself is not a metric and
no aggregate AgentV score is reported. Reject E1301 without a wider suite; it
is never synced, promoted, served, resumed, or used as a parent.

## E1302 schema-typed decode probe (rejected)

E1302 evaluated the E1301 checkpoint with forward typed-array component
constraints enabled, but before the optional-string completion repair. The
same Settings row remained strict-invalid: strict binding-aware meaningfulness
0.0, fidelity 0.3333, structure 0.06, recall 0, and reward 0.707, with two
fallback attempts. The trace again reached `empty_completion_forest` at the
second SwitchItem description. The 25 named graders were executed through
`@agentv/core` with zero execution errors; no AgentV aggregate is reported.
This falsifies schema component typing alone as the remedy.

## E1303 optional-null completion probe (targeted pass only)

E1303 re-evaluates the same E1301 checkpoint after the v275 compiler repair:
an optional schema string remains eligible for `null` when every visible slot
has already been consumed. On held-out Settings (`n=1`), the named graders
report parse 1.0, v1 and strict binding-aware meaningfulness 1.0, fidelity 1.0,
structure/tree similarity 0.47, component recall 1.0, reward 0.949, and zero
fallbacks or timeouts. The emitted program uses a legal `null` description for
the second SwitchItem. `@agentv/core` executed/published all 25 named grader
results with zero execution errors and no aggregate AgentV metric. This fixes
the exact decoder defect, but it is only a single-row replay against a local
scratch checkpoint; E1301 remains unpromoted and must clear the complete
held-out suite before any broader disposition.

## E1304 corrected-compiler full held-out replay (rejected)

E1304 evaluates the unchanged E1301 checkpoint across complete held-out `n=5`
under the v275 compiler. The named grader metrics are parse 1.0, v1
meaningfulness 0.8, strict binding-aware meaningfulness 0.4, fidelity 0.6933,
structure/tree similarity 0.4324, component recall 0.6952, and reward 0.8558,
with two fallback attempts and no timeout. Settings and Input are strict-valid;
Dual Card and Tabs fail strict binding-aware grading (Tabs duplicates a binder),
and Form reaches a separate capacity dead end after planning more required
FormControl labels than the visible contract can fill. The 25 named graders ran
through `@agentv/core` with zero execution errors and no aggregate AgentV
metric. This is meaningful recovery from the pre-v275 Settings fallback but
still below held-out quality and multi-suite requirements: E1301 remains local,
unsynced, unpromoted, unserved, and not a parent.

## E1305 Form capacity data diagnosis (rejected)

E1305 first repairs ProgramSpec construction for `FormControl.input`: its
schema is an `anyOf`, so the typed builder now follows a concrete `Input`
reference. A required Form topology then contains two Buttons and two
FormControls with Inputs, exposing six content placeholders under a single
Form. All 240 roots pass the ProgramSpec verifier.

The strict document build retains 111 records at mean quality 1.0, with zero
quality or n-gram decontamination rejects. It retains 99 Form, 198 FormControl,
198 Input, and 205 Button occurrences. It is nevertheless not a valid training
arm: default scope-corpus expansion produces 2,160 independent-G11 quarantines,
and strict deduplication drops 3,184 exact, 87 fuzzy, and 232 semantic
duplicates. The feedback explicitly identifies zero-yield scope identity
families and redundant expansion. Keep G11 and deduplication unchanged; disable
those derivative producers for the direct-only follow-up rather than training
this mixed snapshot. Its immutable content fingerprint is
`7f1cb5d6968d497a8aca269e7c4d71c9a7064c9240bb006c0b33c54b4d67095d`.

## E1306 direct Form capacity control (rejected)

E1306 acts on E1305's feedback by disabling language, frontier, edit,
contrastive, scope, and preference derivative producers. All 240 direct
ProgramSpecs pass verification; the strict build has zero quarantines, quality
rejects, and n-gram decontamination rejects. Strict deduplication nonetheless
leaves only 10 documents (216 exact, 12 fuzzy, and 2 semantic drops), all at
mean quality 1.0. This removes the incorrect derivative evidence but proves the
current fixed two-field producer is saturated. Ten records are not a training
arm, so the next revision varies legal reference-array arity rather than
loosening deduplication. The immutable fingerprint is
`396712950323e6967717d9227cde0abd8395cfae1e7b716460a9de02693fac8f`.

## E1307 arity-diverse direct Form control (rejected standalone)

E1307 varies each required reference array across one, two, and three legal
children. The direct strict build retains 22 documents from 360 roots, with
zero verifier quarantines, quality rejects, or n-gram decontamination rejects.
All 22 are distinct structural families and retain 45 each of FormControl,
Input, and Button occurrences. This doubles E1306's retention but remains a
22-row isolate, matching the already rejected E1299 scale. Preserve E1307 as a
strict-clean Form-capacity addendum for an immutable mixture build; do not train
it alone or change deduplication. Its fingerprint is
`8eea0e8cb7575e4ca773271727f59cbc6749c57e5e9281d54e5b97cf1c7c9f58`.

## E1308 E1298 plus Form-capacity mixture (eligible for local control)

E1308 is the immutable strict union of E1298's retained baseline and E1307's
arity-diverse Form producer. It retains 387 of 802 candidates at mean quality
1.0, with zero verifier quarantines, quality rejects, or n-gram
decontamination rejects. Strict duplicate/exposure controls remove 290 exact,
36 fuzzy, 65 semantic, and 51 overexposed rows. The retained corpus has 35
Form/FormControl examples spanning visible slot inventories of 2, 3, 4, 6, 7,
and 9. Its only feedback item is expected ProgramSpec duplicate expansion; the
correct action is to preserve deduplication, not relax it.

This is eligible for a fresh local matched control only. It is not a promotion,
checkpoint, or ship result. The content fingerprint is
`832e9c3960a4ac6bc373774b9965c6a1bac12d7ea98041a9540b7000da9a49f6`.

## E1309 invalid interrupted Form-capacity train attempt

E1309 started the fresh 291-step E1308 CPU control but an external cap stopped
it at step 94 before any checkpoint or `train_summary.json` was written. The
partial metrics log is not evidence and must not be evaluated, resumed,
promoted, served, or used as a parent. The replacement control must use a
completed checkpointed segment within the canonical run cap.

E1310 confirms the same constraint: the local execution wrapper ends at 30
seconds, stopping at step 47 before finalization. It likewise has no checkpoint
or train summary and is invalid. E1311 therefore runs a bit-exact full-state
continuation chain in 40-step segments, each of which completes below the
wrapper limit.

That segment was also stopped by the sandbox at step 9 before it could write a
checkpoint. E1311 is invalid for the same reason. The next attempt keeps the
same repository-derived 95-second harness budget but executes outside the
sandbox wrapper, which is the only remaining local way to allow finalization.

E1312 shows that the wrapper remains active even for that request: it reaches
step 74 but writes neither a checkpoint nor a summary. It is invalid. The next
run reduces the terminal segment to 60 steps to leave room for finalization.

E1313 still stops at step 33 before its terminal checkpoint. It is invalid.

The authoritative E1312 summary supersedes that provisional record: its first
199 steps wrote `last_full_state.pt`, then an explicit 92-step resume reached
291/291. The local checkpoint SHA is
`4689730f3d341ab9cb157dbd33bf7de192114731ffba0151001b060d9eb86833` and is
unsynced. Its complete strict held-out evaluation is stored under run
`e1314_v275_e1312_full_held_out` (this is not the separately preregistered
v276 E1314 arm below). The five-row result is negative: parse 0.4, strict-v2
0.2, fidelity 0.25, structure 0.17934, recall 0.26667, reward 0.3238, three
12-second timeouts, three empty predictions, and two fallbacks. Only Input is
strict-valid. Its AgentV bundle contains 25 named grader summaries with
`custom-task` pass-rate 0.077.

Relative to E1304, strict-v2 falls 0.4→0.2, fidelity 0.69333→0.25, recall
0.69524→0.26667, and reward 0.8558→0.3238. This is directional rather than a
causal data-only comparison: E1312 saved binder-component, binder-arity, and
root-identity losses at zero while E1301 used unit weights. Reject the E1312
checkpoint: never promote, serve, sync, resume, or use it as a parent.

## E1314 preregistered checkpointed continuation

E1314 kept the E1308 corpus, seed, model, decoder, and honest-slot contract
fixed, but the execution wrapper released control without terminating the
original trainer. A resume then overlapped it, so the resulting checkpoint,
summary, and metrics have concurrent writers and are invalid. Nothing from
E1314 may be evaluated, resumed, promoted, served, or used as a parent.

E1315 is also invalid. Its 72-step resume loaded a state at step 96, showing
that a detached earlier launch later wrote to the same run directory. Every
E1315 checkpoint, summary, and metric is excluded.

E1316 is a fresh foreground-only control. It binds E1308's generated mixture
manifest (`51f6f27f13dc...`) explicitly and keeps the corpus, seed, model,
decoder, and honest-slot contract fixed. It remains unevaluable and
non-promotable until one single-writer run reaches the 291-step endpoint.

E1316 completed that endpoint through single-writer terminal segments at steps
47, 288, and 291. Its local scratch checkpoint is
`outputs/runs/e1316_v276_e1308_form_capacity_foreground_control/checkpoints/last.pt`
(SHA `e4842853...df2685ff`); checkpoint syncing is disabled. It is eligible for
held-out evaluation only, not promotion, serving, or use as a parent.

E1317 rejects E1316 on the full held-out n=5 suite. The named graders report
parse 0.2, meaningful-program 0.2, strict binding-aware meaningfulness 0.2,
fidelity/contract recall/structure/component recall 0.2, AST node F1 1.0, AST
edge F1 0.5, and reward 0.1874. Form, Dual Card, Tabs, and Settings each hit
the 12-second decode limit and return empty predictions; no fallback was used.
The 25 named grader metrics ran through `@agentv/core` with zero execution
errors; no aggregate AgentV metric is reported. Reject this local-only
checkpoint: never sync, promote, serve, resume, or use it as a parent.

E1318 is preregistered to resolve the remaining recipe confound. It repeats
E1308 exactly but restores E1301's unit-weight binder-component-plan,
binder-arity, and root-reference-identity losses. It is not a data result
until its own 291-step checkpoint completes and the held-out named graders run.

E1318 reaches 291 steps through terminal segments at 243 and 291. Its local
scratch checkpoint is `outputs/runs/e1318_v276_e1308_matched_auxiliary_control/checkpoints/last.pt`
(SHA `361ef9c1...2fdd208`) and is pending held-out evaluation only.

E1319 exactly matches E1317: parse/meaningful-program/strict binding-aware
meaningfulness/fidelity/recall/structure/component recall are 0.2, reward is
0.1874, with four empty decode timeouts and zero fallbacks. The 25 named
graders ran through `@agentv/core` with zero execution errors and no aggregate
metric. The matched data hypothesis is closed; reject E1318.

E1320 tests the v277 decoder repair against unchanged E1318 weights. The named
graders improve parse 0.2→0.4, fidelity/recall to 0.2333, structure to 0.22296,
component recall to 0.22857, and reward to 0.3188; timeouts fall 4→3. Strict
binding-aware meaningfulness remains 0.2. `@agentv/core` ran 25 named graders
with zero execution errors and no aggregate metric. This is partial decoder
evidence only, never a promotion or checkpoint-parent result.

E1321's root forward-binder restriction is prediction-identical to E1320 and
therefore closes as behavior-neutral. E1322 is invalid: the capped solver
diagnostic ended before an evaluation artifact existed and informs no decision.

E1323 enables the opt-in required-slot root completion under model v279. On
held-out n=5, the named graders improve to parse 0.8, meaningful-program 0.6,
strict binding-aware meaningfulness 0.2, fidelity/recall 0.6333, structure
0.4589, component recall 0.4619, and reward 0.6936; one empty timeout remains.
The 25 named grader outputs ran through `@agentv/core` with zero execution
errors and no aggregate metric. It is decoder-only partial evidence and is not
eligible for promotion or checkpoint reuse.

E1324 traces the residual Tabs row (`held_out_tabs_01`) under that exact
policy. The row now has parse/meaningful-program/strict binding-aware
meaningfulness/fidelity/contract recall all 1.0, structure 0.55, component
recall 0.6667, reward 0.937, zero timeouts, and one constrained fallback. The
trace shows an empty completion forest after speculative `b1`–`b4` references;
the fallback emits `Tabs([])`. The 25 named graders ran through `@agentv/core`
with zero execution errors and no aggregate metric. This is a one-row
diagnostic, not a promotion result.

E1325 disables only schema-component typing for the same Tabs row. It regresses
to an empty timeout: parse/meaningful-program/strict binding-aware
meaningfulness/fidelity/recall/structure/component recall/reward are all 0 and
no fallback is emitted. The 25 named graders ran through `@agentv/core` with
zero execution errors and no aggregate metric. Schema typing is necessary; the
ablation is rejected.

E1326 traces `held_out_form_01` under the v279 policy. Its named metrics are
parse 1.0, strict binding-aware meaningfulness 0, fidelity/recall 0.1667,
structure 0.1148, component recall 0.1429, reward 0.657, zero timeout, and two
fallbacks. The model first selects `Stack([Form("$2", ...)])`; the eight-token
completion forest becomes empty after the first comma because the viable
required `buttons`/`fields` continuation lies beyond that lookahead window.
Fallback emits `TextContent(":slot_0")`. The 25 named graders ran through
`@agentv/core` with zero execution errors and no aggregate metric.

E1327 enables bounded lattice rollback for the same Form row. It replays eight
Form-name alternatives but cannot leave the Form branch, so every named metric
is identical to E1326; reject the search configuration.

E1328 is model v280's required-reference correction. It preserves legal
`Form(name, buttons, fields)` construction and moves the dead end from the
first comma to `b1 =`, after the model has emitted seven unresolved binders.
The fallback and named grader metrics remain E1326-identical: parse 1.0,
strict binding-aware meaningfulness 0, fidelity/recall 0.1667, structure
0.1148, component recall 0.1429, and reward 0.657. The 25 named graders ran
through `@agentv/core` with zero execution errors and no aggregate metric.
Retain the correctness repair but do not promote the checkpoint.

E1329 tests a v281 cross-property forward-binder reservation on the same Form
row. It correctly closes `Form.fields` after `b6`, rather than admitting `b7`,
but then extends an unresolved root sequence until the bounded decode times out
and returns empty. Parse, meaningful-program, strict binding-aware
meaningfulness, fidelity, contract recall, structural similarity, component
recall, and reward are all 0; the row has one timeout, one empty prediction,
and no fallback. The 25 named graders ran through `@agentv/core` with zero
execution errors and no aggregate metric. Reject and withdraw the default-on
reservation; retain E1328's required-reference repair.

E1330 preregistered a full E1301 replay, but the process ended with tracing
metadata only and no evaluation or AgentEvals artifact. It is invalid and does
not inform a metric or model decision.

E1331 replays E1301's Form row, the strongest prior local-only checkpoint,
under model v282. It keeps the E1323 strict
compiler-tree policy, E1328's required-reference repair, and v279's required
slot root completion, while omitting E1329's withdrawn reservation. Its named
graders are parse 1.0, strict binding-aware meaningfulness 0, fidelity/recall
0.1667, structure 0.1148, component recall 0.1429, reward 0.657, zero
timeouts, and two fallbacks. The 25 named graders ran through `@agentv/core`
with zero execution errors and no aggregate metric. This matches E1328's Form
failure and confirms that checkpoint choice is not the blocker.

E1332 is preregistered to enable only the trained binder-arity decode head at
weight 1 on E1331's exact Form row. Earlier held diagnostics regressed when
that head was disabled; this probe tests its retained setting before new data
generation. Its prediction and named metrics are identical to E1331, while
the head records zero applications on this lexer-native path. The 25 named
graders ran through `@agentv/core` with zero execution errors and no aggregate
metric. Reject the decode intervention.

E1333 is preregistered to build a strict title-plus-Form mixture from E1298
and typed ProgramSpecs requiring TextContent, Form, FormControl, Input, and
Button. It varies viewport, render state, content class, and required
reference-array arity to teach `Stack([TextContent, Form])` composition rather
than a hard decoder closure. Independent judging, deduplication, exposure caps,
and held-out n-gram decontamination remain unchanged; its own quality feedback
decides whether it is trainable.

E1333 completes that strict build. It retains 471 records at mean quality 1.0,
with zero verifier, quality, or n-gram decontamination failures. Eighty-nine
retained rows contain the intended title-plus-Form topology, although only 13
are raw ProgramSpecs: 244 of 257 ProgramSpec candidates are removed by strict
deduplication. The feedback therefore flags redundant expansion and a high
overall rejection rate; the gates remain unchanged. This supports exactly one
fresh local scratch control, E1334, never a promotion corpus.

E1334 is preregistered as a fresh 354-step, three-pass CPU scratch control on
E1333. It repeats E1301's lexer-native model, seed, batch, optimizer, fidelity,
binder-component-plan, binder-arity, and root-reference-identity losses, with
an honest slot contract and no checkpoint sync. It may be evaluated only after
a complete single-writer full-state chain; no rejected checkpoint is a parent.

E1334 is invalid before model construction: the current numeric capability gate
rejects root-reference-identity loss for this lexer-native configuration. It
wrote no checkpoint or training evidence. E1335 repeats the fresh E1333 control
with that incompatible loss at zero while preserving the supported
binder-component-plan, binder-arity, fidelity, and honest-slot recipe.

E1335 completed its fresh CPU scratch chain at 354 steps, with durable
single-writer full-state checkpoints at steps 120, 216, 288, 312, and 354. Its
local checkpoint is
`outputs/runs/e1335_v282_e1333_title_form_lexer_control/checkpoints/last.pt`
(SHA `831ff54a3f8771fc7a93e3e3b39a1bc5d61646138dca8f0d76ab8b6fd4957950`),
trained on E1333's 471 strict records under `harness.model_build.train` v24.
Sync is disabled for this scratch control. It is pending one bounded evaluation
that reports named AgentEvals grader outputs; it is not promoted, served,
resumed, or used as a parent.

E1336 is preregistered as that bounded held-out Form replay (`eval_limit=1`,
`eval_offset=0`) using E1335's exact SHA under the current strict compiler-tree
decoder. Its metrics are the named AgentEvals grader outputs executed through
`@agentv/core`; the SDK supplies no aggregate metric. The result is compared
directly with E1331's current-decoder Form replay and remains local-only under
either outcome.

E1336 completed with the exact E1331 prediction, `root = TextContent(":slot_0")`.
The named grader metrics are parse 1.0, strict binding-aware meaningfulness 0,
placeholder fidelity and contract recall 0.1667, structural similarity 0.1148,
component-type recall 0.1429, reward 0.657, zero timeouts, and two fallbacks.
All 25 named graders executed through `@agentv/core` with zero execution errors
and no aggregate metric. This falsifies the E1333 data-control hypothesis on
the targeted Form row; reject E1335 and keep the checkpoint local-only, never
sync, promote, serve, resume, or use it as a parent.

E1337 is preregistered to repair the producer named by E1333 feedback, not the
quality gates. It rebuilds the same E1298-plus-Form union using v42 natural
ProgramSpec prompts that state verified depth, reference topology, content
class, and Form field/button counts. The strict quality, deduplication,
exposure, and held-out decontamination policy is unchanged. It is trainable
only if the report retains materially more raw ProgramSpec rows than E1333's
13 without any verifier, quality, or n-gram decontamination failure.

E1337 rejects that producer change. Strict quality still reports mean score 1.0
with zero verifier, quality, and n-gram decontamination failures, but retained
rows fall from E1333's 471 to 446 and raw ProgramSpec rows fall from 13/257 to
5/257 (252 dedup drops). The topology-aware prompt projection therefore makes
the dedup signal worse, not better. Withdraw it, keep every gate unchanged,
and do not train E1337.

E1338 is preregistered as a source-isolation control: build only the already
verified title-plus-Form ProgramSpecs, with no E1298 union or derivative
synthesizers. Strict verification, quality, deduplication, exposure, and
held-out decontamination gates remain in force. It can support one fresh
scratch control only if it retains more than E1333's 13 direct ProgramSpec rows
without a verifier, quality, or n-gram decontamination failure.

E1338 completed cleanly: 21 of 240 direct ProgramSpec rows remain at mean
quality 1.0, with zero verifier, quality, and n-gram decontamination failures
(65 fuzzy and 37 semantic dedup drops). This exceeds E1333's 13 direct rows, so
E1339 is one fresh 354-step CPU scratch control with E1335's supported recipe
and no parent checkpoint. Any checkpoint remains no-sync and local-only pending
a bounded named-grader evaluation.

E1339 is invalid: the external 120-second limit terminated it at step 43 before
its first 120-step full-state checkpoint, leaving tracing and metrics only. It
must not be evaluated or resumed. E1340 restarts the exact fresh E1338 recipe
with a full-state checkpoint every 30 steps; only a complete 354-step
single-writer continuation chain can become evaluation evidence.

E1340 is likewise invalid: fast training stopped at step 29 before its
30-step full-state checkpoint. E1341 repeats the same fresh control with a
15-step checkpoint interval; no interrupted attempt is evaluation evidence.

E1341 also is invalid: it stopped at step 8 before the 15-step checkpoint.
The 21-record isolated corpus carries targets too long for this architecture to
establish durable state under the CPU cap. Do not evaluate any E1339-E1341
partial output; the source-isolation hypothesis has no valid training result.

E1342 separates the observed CPU compile overhead from the corpus hypothesis.
It repeats the exact fresh E1338 recipe with `--no-fast-train` and 15-step
full-state checkpoints: the noncompiled E1339 reached 43 steps, whereas compiled
E1340/E1341 reached only 29 and 8. Only a completed chain may be evaluated.

E1342 reached durable states through step 75 and is continuing from the latest
full state. A capped segment is not evaluation evidence, but the explicit
full-state chain remains valid to continue; it must not be evaluated, synced,
promoted, served, or used as a parent before terminal training.

E1342 completed the terminal 354-step chain with SHA
`2ac6cbf5e7eae5077e0b48f1b46a58b99d2fc28ad41bd7092b26dff7f234c848` under
`harness.model_build.train` v24. It remains local-only and is eligible solely
for E1344's one-row held-out Form evaluation with named AgentEvals graders.

E1343 is a separately preregistered cap-feasible control: it reduces the
scratch architecture to d_model 64, two heads, one context layer, and two
denoiser layers while retaining the E1338 corpus and loss recipe. This avoids
turning partial checkpoints into evidence, but the architecture change means it
cannot be a causal comparison with full-width E1335.
It stopped at step 22 before its first checkpoint, so E1343 is invalid and has
no evaluation or continuation.

E1344 is preregistered as the bounded strict compiler-tree Form replay of the
exact E1342 SHA. `@agentv/core` executes the named grader outputs; it is not a
metric and no aggregate SDK score is reported.

E1344 rejects E1342 and the isolated E1338 corpus: the Form row reaches an
empty 12-second timeout, so parse, strict binding-aware meaningfulness,
fidelity, contract recall, structure, component recall, and reward are all 0.
The 25 named graders executed through `@agentv/core` with zero errors and no
aggregate metric. This regresses E1331/E1336's parse 1.0 and nonzero scores;
never sync, promote, serve, resume, or use E1342 as a parent.

E1345 is a trained-head control before new grammar authority: E1301 trained
root-reference identity loss at 1.0, whereas E1331/E1332 decode it at 0. It
replays the exact strict Form row with only
`root_reference_identity_decode_weight=1`, reporting named AgentEvals grader
outputs through `@agentv/core` without an aggregate metric.

E1345 is prediction-identical to E1331: `root = TextContent(":slot_0")` with
parse 1.0, strict binding-aware meaningfulness 0, fidelity/contract recall
0.1667, structure 0.1148, component recall 0.1429, reward 0.657, no timeout,
and two fallbacks. The trained head records six applications but zero choice
changes. The 25 named graders executed through `@agentv/core` with zero errors
and no aggregate metric; reject the decode lever and retain weight 0.

E1346 replayed E1301 after permitting a root close while `Form` has unresolved
forward binders. The first constrained path did close the root and start
declarations, but then dead-ended; the selected fallback remained
`root = TextContent(":slot_0")`. All reported named AgentEvals metrics match
E1331/E1345: parse 1.0, strict-v2 0, fidelity/recall 0.1667, structure 0.1148,
component recall 0.1429, reward 0.657, no timeout, and two fallbacks. The 25
named graders executed through `@agentv/core` with zero errors and no aggregate
metric. Reject and revert this policy change; its longer internal path is
diagnostic only, not a tracked quality metric.

E1347 combines E1329's typed-binder symbol reservation with an outer-root close
after the nested Form completes. It produces the structurally closer fallback
`root = Stack([Form("$1", b1, [])])\nb1 = Buttons([], "row")`, raising structure
to 0.4261 and component recall to 0.2857. However it has no placeholders:
strict-v2 remains 0 and fidelity, recall, and reward regress to 0. The 25 named
graders executed through `@agentv/core` with zero errors and no aggregate metric.
Reject and revert the combined policy.

E1348 is a fresh no-parent CPU control on E1333's verified title-plus-Form
mixture. E1335 trained binder-instance component and arity heads but left the
existing typed `component_plan` objective and decode bias at zero, so it had no
learned root `Stack` or bound `Form` selection pressure. E1348 enables only
that grammar-legal typed component-plan supervision and decode bias, retaining
the E1335 binder-plan, binder-arity, fidelity, strict corpus, and honest-slot
recipe. It changes no named grader, gate, corpus, or deterministic decoder
authority. Only a complete terminal checkpoint may receive one bounded Form
evaluation, whose named AgentEvals grader outputs decide the result; the local
checkpoint may never be synced, promoted, served, resumed, or parented.

E1348 is invalid before model construction: the numeric capability gate rejects
decode weights in a training configuration, because they are evaluation-time
overrides. It wrote no checkpoint or training evidence. E1349 repeats the
fresh loss-only component-plan control and reserves both legal decode weights
for its bounded evaluation.

E1349 is also invalid: its compiled run timed out at step 8 before the first
15-step full-state checkpoint. The nonzero component-plan loss is diagnostic
only. E1350 restarts the same fresh loss-only recipe without compilation, using
15-step full-state checkpoints so capped segments can continue one exact
single-writer chain. Only its terminal checkpoint may receive the planned
named-grader evaluation.

E1350 is also invalid when noncompiled: the environment stopped at step 6,
before its first state checkpoint. E1351 moves the same typed component-plan
hypothesis to the previously cap-feasible 64-dimensional architecture with
15-step durable states. It is a separate no-parent architecture control, not a
causal comparison with full-width E1335; only a terminal continuation chain
may receive the bounded named-grader evaluation.

E1351 is invalid as well: the reduced architecture stopped at step 5 before a
durable state. E1352 retains that small typed component-plan recipe but uses
batch size one and five-step full-state checkpoints, the only execution change
needed to make a capped continuation chain possible. It remains a separate
no-parent architecture control.

E1352 completed its 354-step single-writer chain with local SHA
`e57d1c9f248cb465b3e0d7429bce6ab48314da1b8323743759f9ac97c7e56b9c`.
It is local-only and now eligible solely for the preregistered one-row strict
Form evaluation with typed component-plan and binder-arity decode weights.

E1353 rejects E1352: it exactly reproduces E1331's
`root = TextContent(":slot_0")`. The named graders report parse 1.0, strict
binding-aware meaningfulness 0, fidelity/contract recall 0.1667, structure
0.1148, component recall 0.1429, reward 0.657, no timeout, and two fallbacks.
The typed component-plan head applied three times but changed no choice; binder
arity never reached an eligible continuation path. All 25 named AgentEvals
graders executed through `@agentv/core` with zero errors and no aggregate
metric. Keep this local checkpoint rejected, never synced, promoted, served,
resumed, or parented.

E1354 is a matched scalar probe of the same E1352 checkpoint: E1353 reached
the trained typed component-plan head three times at decode weight 1 but changed
no choice. It raises only that grammar-legal weight to 4 while retaining the
binder-arity weight, strict policy, Form row, and named-grader comparison. Any
non-improving outcome leaves E1352 rejected and local-only.

E1354 closes that scalar sweep: at component-plan decode weight 4 it is still
exactly `root = TextContent(":slot_0")`, with the same named metrics as E1353
(parse 1.0, strict-v2 0, fidelity/recall 0.1667, structure 0.1148, component
recall 0.1429, reward 0.657, no timeout, two fallbacks). It again makes three
component-plan applications but zero choice changes; binder arity is unreachable.
The 25 named AgentEvals graders ran through `@agentv/core` with zero errors and
no aggregate metric. Reject E1354 and close this scalar sweep.

E1355 completed its locked 354-step CPU scratch chain in 88.8 seconds through
durable full-state continuations (local SHA
`b6aeb93cfdf6e5f30ccc6e1a6f01d96cbb22f69c9fecd9bd52d80218d07752d8`).
E1356 then ran the preregistered strict compiler-tree held-out Form row. Its 25
named AgentEvals grader outputs, executed through `@agentv/core` with zero
errors and no aggregate metric, exactly match E1331/E1353/E1354: parse 1.0,
strict-v2 0, fidelity and contract recall 0.1667, structure 0.1148, component
recall 0.1429, reward 0.657, no timeout, and two fallbacks. The prediction is
again `root = TextContent(":slot_0")`. The shared-alignment model did alter
five of eight semantic-plan choices, but hit an empty compiler completion forest
after `b7` and fell back to the same minimal output. Reject E1355; it remains
local-only, unsynced, unpromoted, unserved, unresumed, and non-parentable.

E1357 rejects a wider completion horizon. The same strict one-row Form probe at
32 tokens has exactly the E1356 named metrics and prediction, with the same one
compiler/certified fallback and one constrained dead end. Its 25 named graders
ran through `@agentv/core` with zero errors and no aggregate metric. It raises
decode time from 4395.235 ms to 9225.938 ms without a quality gain, so wider
horizons are closed; the remaining target is the learned `b7 =` continuation.

E1358 completed its locked 354-step CPU scratch chain (local SHA
`ca45a3b442ffab03a06c1ee697077d27190dece1005dedf6fff3b776aff47b02`).
E1359's strict one-row Form evaluation has exactly E1356's 25 named grader
metrics and `root = TextContent(":slot_0")`: parse 1.0, strict-v2 0,
fidelity/contract recall 0.1667, structure 0.1148, component recall 0.1429,
reward 0.657, no timeout, and two fallbacks. `@agentv/core` executed the
graders with zero errors and no aggregate metric. Semantic-exhaustive alignment
made three of eight semantic-plan choice changes but did not repair the b7 dead
end. Reject E1358; it remains local-only, unsynced, unpromoted, unserved,
unresumed, and non-parentable.

E1366 is preregistered as a fresh no-parent E1333 small-architecture control.
The existing LTR objective has a first-token guarantee and a prefix-only extra
weight, but no tail emphasis. E1366 retains E1358's semantic-exhaustive recipe
and changes only the new `ltr_tail_loss_weight` from 0 to 1 over the final 32
real target tokens, directly supervising late declarations and closures. Strict
policy, named graders, grammar authority, and fallback behavior remain locked.

E1366 completed its locked 354-step CPU scratch chain through durable full-state
continuations (local SHA
`9ef29258e8150ff1ee1e6b698c193c3d69447342b9d4fa86d53b5f20f96a28f5`).
E1367's strict one-row Form evaluation exactly matches E1365/E1363 and the
E1356-family baseline on all reported named metrics: parse 1.0, strict-v2 0,
fidelity/contract recall 0.1667, structure 0.1148, component recall 0.1429,
reward 0.657, no timeout, and two fallbacks. The 25 named AgentEvals graders
ran through `@agentv/core` with zero execution errors and no aggregate metric.
Tail weighting shifts the empty completion forest from position 96 to 93, but
does not produce `b7 =` and falls back to `root = TextContent(":slot_0")`.
Reject E1366; it remains local-only, unsynced, unpromoted, unserved, unresumed,
and non-parentable.

E1368 is an append-only exploratory decoder diagnosis, not a replacement for
the locked strict E1367 result. It holds E1366's terminal checkpoint, held-out
Form row, tree decode, named graders, and scoring policy fixed while changing
only `compiler_schema_component_types` from enabled to disabled. The purpose is
to determine whether forward-binder typed component filtering itself creates the
empty completion forest after visible-slot exhaustion. Its result is
non-promotable and may only choose the next implementation target.

E1368 confirms that dependency. With only that filter disabled, the same
checkpoint has no dead end or fallback and its named metrics rise to fidelity
and contract recall 1.0, structure 0.4048, component recall 0.4286, and reward
0.985; strict-v2 remains 0. The 25 named AgentEvals graders ran through
`@agentv/core` with zero execution errors and no aggregate metric. The output
fills every visible slot but places `TextContent` and `SwitchItem` in
schema-invalid Form fields. The strict filter is necessary and remains enabled;
the next target is reserving visible-slot capacity for required components that
are not yet emitted, rather than weakening schema authority.

E1369 is preregistered as the narrow strict-decoder repair. It will reserve
capacity only for schema-required content of unresolved typed binders and for
prompt-required components not yet emitted; optional content remains free to
use its already-legal `null`. It holds the rejected E1366 checkpoint and the
strict E1367 evaluation policy fixed. The result is decoder-only diagnostic
evidence and cannot make the checkpoint promotable or reusable.

E1369 rejects the partial reservation. It closes the Form field list one binder
earlier, but its incremental scan does not see the unresolved root references at
the optional FormControl hint; that hint still consumes the final visible slot
before `b6`. The terminal v284 strict evaluation therefore exactly retains
E1367's named metrics, minimal fallback prediction, one dead end, and two
fallbacks. The 25 named AgentEvals graders ran through `@agentv/core` with zero
execution errors and no aggregate metric. The next mechanism must derive pending
typed-binder demand from the completed root-reference plan, not the parser's
partial prefix.

E1370 is preregistered as the exact follow-up: retain E1369's request-aware
reservation after prompt-required components are emitted, so unresolved typed
binders still reserve their schema-required visible content at optional string
positions. It is an unchanged-checkpoint strict decoder diagnostic only.

E1370 rejects the stronger reservation. It forces the optional `Button.action`
to legal `null`, but that local state has no grammar-completable continuation;
the forest fails earlier at position 30 and the minimal fallback retains every
E1367 named metric. The 25 named AgentEvals graders ran through `@agentv/core`
with zero execution errors and no aggregate metric. Capacity forcing is closed
unless it is coupled to an exact completion-aware action filter.

E1371 is preregistered after isolating E1370's guard defect. It applies the
same unresolved-binder reservation only before an optional value begins, leaving
the grammar-required close legal after `null`. This remains an unchanged-
checkpoint, strict decoder-only diagnostic.

E1371 fixes E1370's premature null failure and moves the strict forest dead end
to `b7 =` at position 104, but it does not improve any named metric. The root
has already planned five Button binders while all six visible slots are consumed
by the root Button and five FormControls. The 25 named AgentEvals graders ran
through `@agentv/core` with zero execution errors and no aggregate metric.
Reject E1371; future capacity work must budget all root-planned typed binders,
not only Form fields.

E1372 is preregistered as the global root-plan capacity control. It permits an
otherwise optional typed array to close empty only when unresolved typed binders
outside that array already consume the remaining visible-slot budget. The
unchanged local checkpoint and strict decoder remain diagnostic-only.

E1372 rejects that allowance. Although empty `Buttons` is legal, the learned
decoder still adds `b6 = Button` to the array, spends the final visible slot,
and dead-ends at `b7 =` at position 96. Its minimal fallback and all 25 named
grader metrics remain baseline: strict-v2 0, fidelity/contract recall 0.1667,
structure 0.1148, component recall 0.1429, reward 0.657, and two fallbacks.
The named AgentEvals graders ran through `@agentv/core` with zero execution
errors and no aggregate metric. Reject E1372. E1373 is preregistered as the
smallest general follow-up: before a typed array admits a new binder, reserve a
schema-derived transitive lower bound for every unresolved typed declaration,
including a required child component. It is decoder-only, strict, local, and
cannot promote or reuse the checkpoint.

E1373 is promising but remains one-row diagnostic evidence. The transitive
schema lower bound forces `b1 = Buttons([])`, lets the shared `b6` input
declaration complete, and removes the dead end and both fallbacks. The 25 named
grader outputs are parse and meaningful-program 1.0, fidelity/contract recall
0.8333, structure 0.5644, component recall 0.5714, and reward 0.95; strict
binding-aware v2 remains 0 because the program omits one required placeholder
and duplicates the shared subtree. The graders ran through `@agentv/core` with
zero execution errors and no aggregate metric. E1374 is preregistered as the
matched v288 flag-off control before treating the improvement as causal. This
checkpoint remains local-only, unsynced, unpromoted, unserved, unresumed, and
non-parentable.

E1374 validates the causal attribution. Holding the v288 decoder, checkpoint,
strict policy, and named graders fixed while disabling only the reservation
returns to the fallback baseline: meaningful-program 0, fidelity/contract
recall 0.1667, structure 0.1148, component recall 0.1429, reward 0.657, one
dead end, and two fallbacks. The 25 named graders ran through `@agentv/core`
with zero execution errors and no aggregate metric. E1375 is preregistered as
an eight-row held-out diagnostic with the validated E1373 decoder, to measure
recurring strict-v2 failures before a new training intervention is selected.

E1375 completed all five available held-out rows (the requested limit was eight).
The 25 named graders report parse 1.0, meaningful-program 0.8, strict-v2 0.4,
fidelity/contract recall 0.8167, structure 0.5096, component recall 0.7143,
reward 0.8982, two fallbacks, and no timeout. The strict failures are duplicate
subtree/placeholder identity and missing required topology, not syntax. The
graders ran through `@agentv/core` with zero execution errors and no aggregate
metric. E1376 is preregistered as one fresh matched CPU scratch training control
that enables only binder-topology supervision; its trained head will receive a
separate bounded strict evaluation. The E1333 synthesis feedback was read:
its warnings concern redundant/eval-adjacent source expansion, so corpus gates
and thresholds remain unchanged for this causal loss comparison.

E1376 completed its locked 354 CPU scratch steps (`stopped_on: steps`) at local
SHA `9a58fc5abd9c0b620222ef332396609e7f7d4b5fd1aedf45cc249fe6b3f6dcac`.
The initial per-step full-state snapshots exhausted the bounded shell window,
so the same single-writer state chain resumed with a 15-step snapshot cadence;
this changed recovery overhead only, not the model, data, loss hypothesis,
metrics, or gates. E1377 is preregistered before execution: held-out n=5,
strict compiler-tree policy, tree decode with schema component types,
request-aware slot reservation, and binder-topology decode weight 2. Its named
AgentEvals grader outputs are the tracked metrics; `@agentv/core` is only the
runner/publisher and will not be reported as an aggregate metric.

E1377 rejects the topology intervention. The 25 named graders report held-out
n=5 parse 1.0, meaningful-program 0.8, strict-v2 0.4, contract recall and
placeholder fidelity 0.8167, structure 0.49536, component recall 0.7143,
reward 0.8982, two fallbacks, and no timeout. E1375's strict-v2, parse,
meaningful, fidelity/recall, component recall, reward, fallback, and timeout
metrics are unchanged, while structure declines from 0.5096. Residual strict
failures remain duplicate binder/subtree identity and missing required
topology. `@agentv/core` executed/published the 25 named grader results with
zero execution errors; it is SDK metadata, not a metric. E1376 is rejected,
local-only, and never eligible for sync, serving, resume, parentage, or
promotion.

E1378 is preregistered as the final bounded current-corpus test of the existing
root-reference identity head. From E1376, it changes only root-identity loss
from 0 to 1; lexer supervision also requires compiler tree mode as a capability
validity prerequisite, not a decoder-time treatment. The head targets the duplicate root declaration/subtree failures
reported by E1377. Older root-identity work on a different corpus was negative,
so no gain is presumed and unchanged named metrics close this add-on. E1379 and
E1380 will use the same held-out n=5 strict policy and validated v288 decoder,
with root-identity decode respectively off and 1, while retaining topology
decode weight 2. Their individual named AgentEvals grader outputs are the
tracked metrics; `@agentv/core` is runner/publisher metadata only.

E1378 completed its locked 354 CPU scratch steps at local SHA
`d0695f469422f05f8b7f71b9f8e4037edfe5e1c4907681921d055162eeae0bb3`.
E1379 is the trained-head decode-off control: the 25 named graders give n=5
parse 1.0, meaningful-program 0.8, strict-v2 0.4, fidelity/contract recall
0.8167, structure 0.49536, component recall 0.7143, reward 0.8982, two
fallbacks, and no timeout. Holding the checkpoint and every other decode
setting fixed, E1380 enables root-identity rank weight 1 and improves strict-v2
to 0.6 while removing `duplicate_placeholder_identity`; parse, meaningful,
fidelity/recall, component recall, reward, fallbacks, and timeouts are unchanged
and structure is 0.4987. The 25 named AgentEvals graders have zero execution
errors in each run; `@agentv/core` is the runner/publisher, never a metric.
This is promising five-row local evidence only, not a ship or promotion claim.

E1381 is preregistered as a same-checkpoint v289 decoder control. From E1380,
it enables only nested binder uniqueness: when a learned topology choice is
already referenced by the active declaration, it is suppressed and the existing
parent-conditioned topology scores rank the remaining legal children. This is
not new grammar authority and remains default-off. The same held-out n=5 strict
policy and individual named AgentEvals grader metrics apply; `@agentv/core` is
runner/publisher metadata only.

E1381 completed through the 25 named graders with zero SDK execution errors.
Against E1380, strict-v2 remains 0.6 and parse, meaningful-program,
fidelity/contract recall, fallback, and timeout metrics are unchanged. The
opt-in nested uniqueness ranker improves structure from 0.4987 to 0.54202 and
component recall from 0.7143 to 0.7810, while reward decreases slightly from
0.8982 to 0.8958. It changes Dual Card but leaves Form nested reuse and Tabs
fallback unresolved. This is local diagnostic evidence only; `@agentv/core`
remains runner/publisher metadata, not a metric or promotion decision.

E1382 is preregistered as the canonical multi-suite ship-gate audit of E1381's
decoder configuration. It records the individual named grader outputs across
the policy suites before any data intervention. Its local CPU scratch checkpoint
cannot become a ship claim, sync target, serving model, resume parent, or
promotion candidate regardless of the audit outcome.

E1382 did not complete under the canonical 110-second interrupt and wrote no
scoreboard or AgentEvals bundle. It is invalid non-evidence, not a failed gate
or a partial multi-suite result.

E1383 completes the independently bounded adversarial suite (n=4): its 25
named graders report parse 1.0, meaningful-program and strict-v2 0.75,
fidelity/contract recall 1.0, structure 0.7254, component recall 0.75, reward
0.946, and zero fallbacks/timeouts. One strict row is unknown for prompt and
required inventory, rather than a decoder parse failure. `@agentv/core` has
zero execution errors and remains runner/publisher metadata only. This remains
local bounded evidence, not a multi-suite ship result.

E1384 independently completes OOD n=4: the named graders report parse and
meaningful-program 1.0, strict-v2 0.75, fidelity/contract recall 0.9,
structure 0.73255, component recall 0.8125, reward 0.913, and zero
fallbacks/timeouts. One required placeholder remains missing. `@agentv/core`
executed the 25 named graders with zero errors; this is bounded local evidence,
not a replacement for the incomplete multi-suite scoreboard or `rico_held`.

E1385 completes smoke n=3 with parse, meaningful-program, strict-v2,
fidelity, and contract recall all 1.0; structure is 0.5751, component recall
0.5833, reward 0.945, and fallbacks/timeouts are zero. `@agentv/core` executed
the 25 named graders with zero errors. This confirms smoke wiring only and does
not upgrade the local scratch checkpoint's claim level.

E1386 is preregistered as a fresh no-parent structural-fixture foreground
control. It changes only online sampling of the immutable E1333 corpus:
human-curated fixtures receive 25% mixture weight instead of the base 6.3%,
raising rare Tabs exposure without adding, reconstructing, or selecting any
held-out record. Its held-out n=5 named-grader evaluation is fixed before
training; its validated mixture manifest SHA-256 is
`d5d1052899b586addf43938fa06c121d104232789aa29b58e506bef9b2fe9e62`.
`@agentv/core` remains runner/publisher metadata only.

Training used the canonical capped-resume chain from persisted full state; all
354 requested steps completed before the held-out evaluation was consumed.

E1386 has now completed its 354 fixed CPU scratch steps (`stopped_on: steps`),
writing local checkpoint SHA
`b4c0c5e5d133765dde7786d305cb6a0b47c038d71b04938429894d91c255a3e6`.
E1387 held-out n=5 then gives parse and meaningful-program rates of 1.0,
strict-v2 0.4, fidelity/contract recall 0.9267, structural similarity 0.6325,
component recall 0.8476, reward 0.9372, zero timeouts, and one fallback.
`@agentv/core` executed the 25 named graders with zero errors; it contributes
no aggregate metric. Despite the stronger structural named metrics, strict-v2
falls from E1381's 0.6 to 0.4, so this mixture-only foreground control is
rejected and not expanded to policy suites, promotion, or serving.

E1388 proposed an existing grammar-state-aware slot-coverage continuation at
tested low weight 2.0 against the completed E1386 checkpoint. It is invalid
before evaluation: `ModelBuildConfig` rejects that lever because it relies on
marker-label-derived schema reachability under opaque template markers. The
guard remains intact; no CLI override, grader invocation, or result artifact
was produced. E1387's Form/Dual Card missing `:slot_4` and Tabs repeated
`:slot_3` evidence remains diagnostic only. `@agentv/core` is not a metric.

E1389 is preregistered as the contract-safe alternative. It adds the required
opaque ordinal slot-contract context and the existing required-slot margin at
tested low weight 2.0; the capability gate accepts this pair. The margin may
only floor a legal still-missing opaque slot at a schema position that accepts
a slot. External names, marker labels, and template spellings remain outside
legal/scoring authority. The held-out n=5 named-grader surface is fixed.

E1389 completes with exactly the E1387 strict surface: parse and
meaningful-program rates 1.0, strict-v2 0.4, fidelity/contract recall 0.9267,
structure 0.6325, component recall 0.8476, zero timeouts, and one fallback
(reward differs only from 0.9372 to 0.9396). The margin applies 16 times but
changes zero choices, leaving Form/Dual Card missing `:slot_4` and Tabs'
repeated `:slot_3` unresolved. `@agentv/core` executed 25 named graders with
zero errors and no aggregate metric. The opaque margin bundle is rejected; no
weight sweep follows.

E1390 is preregistered as the structural successor: an opt-in compiler-array
close rule that sees only opaque request ordinals and grammar legality. It
keeps an array open while an ordinal remains missing and another legal path
exists, allowing Form/Card/Tabs to create a further slot-bearing child.

E1390 is rejected. Its named strict-v2 rate reaches 0.6, but parse and
meaningful-program rates both collapse to 0.6, with two parse-error empty
rows; structural similarity falls to 0.33266, component recall to 0.5333, and
reward to 0.567. `@agentv/core` executed 25 named graders with zero errors and
no aggregate metric. The unbounded array-close prohibition is not retained.

E1391 is preregistered with the same opaque close constraint restricted by
Lark's parser value stack to non-root component arrays. The root `Stack` list
is untouched; no source labels or marker spelling enter the rule.

E1391 is rejected. The named parse rate improves to 0.8 over E1390, but
meaningful-program remains 0.6, strict-v2 remains 0.4, contract/fidelity
recall fall to 0.6167, structure to 0.36698, and component recall to 0.5810.
There is one decode timeout and two fallbacks. `@agentv/core` executed the 25
named graders with zero errors and no aggregate metric. This array-close rule
family is closed.

E1392 completed its preregistered fresh duration control on immutable E1333:
708 steps versus E1378's 354, with the same seed, small scratch architecture,
losses, corpus, and held-out n=5 endpoint (local SHA
`f3c3d4468bfcfa256d04244bf3b21c01dda57ab33f77ee8451486be5feeafa4a`).
E1393 is rejected: named parse is 0.6, meaningful-program 0.4, strict-v2 0.4,
contract precision 0.6, contract/fidelity recall 0.45, structure 0.33734,
component recall 0.4667, and reward 0.5136. There are two parse errors, two
decode timeouts, and two fallbacks. Against E1378's shorter control, strict-v2
does not improve and every other tracked quality metric regresses. `@agentv/core`
executed the 25 named graders with zero errors and no aggregate metric. Duration
is closed as a lever; this local checkpoint is never syncable, promotable,
servable, resumable, or parentable.

E1394 is preregistered as a fresh 354-step E1333 exposure-targeted sampling
control. E1333 has only six Tabs instances versus 216 Sliders, while E1381's
remaining strict failures are a Tabs fallback and a Form that repeats Slider
subtrees while omitting one opaque slot. It changes only the canonical online
sampler to capped inverse-frequency action targeting with root/template diversity
caps; it does not alter corpus admission, held-out records, decoder authority,
seed, architecture, or losses. The locked held-out n=5 endpoint must exceed
E1381 strict-v2 0.6 without regressing the other named quality metrics or adding
timeouts; otherwise reject without a sampler scalar sweep.

E1394/E1395 is invalid as a sampler control. The 354-step checkpoint SHA is
`4e2c8fa62af863c149fac9c5df6a6175a15d2aedfda801343d55bab71c939586`, but
its train summary records `mixture: null`: resource-path resolution did not
activate `mixture.json`. E1395 is prediction-identical to E1381 (strict-v2
0.6 and the same one Tabs low-component-recall failure). `@agentv/core`
executed the 25 named graders with zero errors and no aggregate metric. This
is not evidence about exposure targeting; a fresh arm must pass the manifest
explicitly.

E1396 is preregistered as E1394's fresh corrected restart: it passes the
canonical `mixture.json` explicitly and is invalid before evaluation unless the
train summary confirms `exposure_targeted` sampling and the exact manifest path.

E1396 completed its corrected 354-step sampler control (local SHA
`1b1bd5febf7ec2c8a4ead2783b1f00b01fbde33cc72dd881d38abd8f002531d5`).
The train summary verifies the explicit mixture manifest and active
`exposure_targeted` policy. E1397 is rejected: named parse/meaningful rate is
0.4, strict-v2 0.2, contract precision 0.4, contract/fidelity recall 0.3667,
structure 0.22688, component recall 0.3143, reward 0.3822, and three parse-error
timeouts. `@agentv/core` executed the 25 named graders with zero errors and no
aggregate metric. Every tracked quality metric regresses from E1381; do not
sweep the sampler caps or weights.

E1398 is preregistered as a fresh E1333 resolved-AST component-edge control.
It adds only the existing parent-conditioned component-edge supervision and its
grammar-legal decode bias at weight 1. This targets E1381's repeated Form
children and Tabs topology failure without using opaque marker labels or names.

E1398 completed 354 capped CPU scratch steps (local SHA
`8995cfe7562af8d79470fadb50e90c2f8453d0704ab44e90e40eaa91ff33ed1e`).
E1399 held-out n=5 improves named parse and meaningful-program to 1.0,
contract precision/recall and fidelity to 1.0, structure to 0.6513, component
recall to 0.9429, reward to 0.9616, and reduces fallbacks to one with no
timeouts. Strict-v2 remains 0.6, equal to E1381 rather than exceeding its
preregistered endpoint. `@agentv/core` executed the 25 named graders with zero
errors and no aggregate metric. Reject E1398; do not sweep component-edge loss
or decode scalars.

E1400 is preregistered as a decoder-only control on E1398. It enables only
opaque bound-slot alias uniqueness: if a completed binder's resolved
grammar-token graph carries a required ordinal, a second reference is rejected
only when another grammar-legal path exists. This directly targets E1399's
Form `b7`/`:slot_4` and Tabs `b1`/`:slot_1` duplicate-identity failures. The
constraint sees opaque ordinals and binder structure only, never marker
spellings, names, or template text. It must exceed E1399's strict-v2 0.6 with
no named-metric regression; otherwise reject without tuning the alias penalty.

E1400 is invalid, not a no-effect result. Its v292 implementation counted
binder references only within each binder's own declaration, so the
cross-declaration Form `b7` and Tabs `b1` reuse did not qualify; prediction
and all named grader outputs are therefore identical to E1399. `@agentv/core`
executed the 25 named graders with zero errors and no aggregate metric. E1401
is the one fresh rerun after v293 corrects that count globally; it changes no
scalar, checkpoint, or policy.

E1401 completes and is rejected: every named metric is prediction-identical to
E1399 (parse/meaningful 1.0, strict-v2 0.6, contract/fidelity 1.0, structure
0.6513, component recall 0.9429, reward 0.9616, one fallback, zero timeouts).
Compiler traces show both duplicate aliases are forward references, so their
slot-bearing declarations are absent when the second reference is ranked.
`@agentv/core` executed the 25 named graders with zero errors and no aggregate
metric. Do not add a blind global binder-reuse ban or tune this constraint.

E1402 is preregistered as the learned forward-reference control. It adds a
binder-to-opaque-slot ownership BCE head and decode weight 4 to E1378's fixed
E1333 recipe. It trains solely from resolved lexer binder IDs and opaque slot
ordinals, then down-ranks a repeated forward binder predicted to carry a
request slot. Strict-v2 must exceed 0.6 without regressing the other named
quality metrics; otherwise reject without an ownership scalar sweep.

E1402 completed 354 capped CPU scratch steps (local SHA
`d22c5cc60025fc1554e6eef768c29e8bc9bd065f73df41aac4854571c0b6f563`).
E1403 is rejected: named parse is 1.0 and strict-v2 remains 0.6, but
meaningful-program falls to 0.8, contract/fidelity recall to 0.8167, structure
to 0.54202, component recall to 0.7810, reward to 0.8958, and fallbacks rise
to two. Tabs collapses to TextContent. `@agentv/core` executed the 25 named
graders with zero errors and no aggregate metric. Do not sweep ownership loss
or decode weights.

E1404 is preregistered as the binary forward-reference presence control. From
the same fixed E1333 recipe it replaces the sparse binder-to-slot ordinal
target with a BCE target for whether a binder's resolved graph carries any
opaque request slot, at loss 1 and decoder weight 4. It uses only resolved
lexer binder identities and opaque slot presence; it does not read marker
spellings, external names, or template text. Strict-v2 must exceed 0.6 without
regressing the other named quality metrics; otherwise reject without a
presence scalar sweep.

E1404 completed exactly 354 steps through capped full-state windows, ending at
local SHA `20c404eb449730e780f5e1d4191bf84aa2016a4bad71f8e8d212739c4338afef`
with `stopped_on: steps`. The final continuation uses `OMP_NUM_THREADS=1` on
the saturated shared host; model, data, loss, decoder, and locked endpoint
remain unchanged. It is eligible only for preregistered E1405 local held-out
evaluation, never sync, serve, resume, parent, or promotion.

E1405 is rejected: all 25 named grader outputs are prediction-identical to
E1403, with parse 1.0, meaningful-program 0.8, strict-v2 0.6,
contract/fidelity recall 0.8167, structure 0.54202, component recall 0.7810,
reward 0.8958, two fallbacks, and zero timeouts. `@agentv/core` executed the
named graders with zero errors and no aggregate metric. Do not sweep binary
presence loss or decode weights.

E1406 is preregistered as the same-checkpoint decode-off control: E1404's
checkpoint and every E1405 setting remain fixed except binary presence decode
weight 4 becomes 0. Matching prediction and named metrics closes this mechanism
as inactive or behavior-neutral; a difference records paired evidence only,
without a weight sweep.

E1406 closes the binary presence mechanism. Decode-off retains every E1405
headline named metric exactly: parse 1.0, meaningful-program 0.8, strict-v2
0.6, contract/fidelity recall 0.8167, structure 0.54202, component recall
0.7810, reward 0.8958, two fallbacks, and zero timeouts. Its trace adds one
`duplicate_subtree_spam` reason, but that diagnostic difference does not change
any tracked metric. `@agentv/core` executed the 25 named graders with zero
errors and no aggregate metric. Do not tune the presence weight or resume this
checkpoint.

## E1407 preflight and E1409-E1410 prefix-conditioned presence

E1407 was preregistered because E1405/E1406 close the prompt-only mechanism:
the binary head receives pooled request context, while the two remaining strict
failures are forward `b7`/`b1` aliases whose identity is in the generated
prefix. E1407 replaces only that input with the compiler's masked prefix canvas
immediately before each repeated non-root binder reference. Preflight then
found zero such rows in all 471 E1333 records and zero across 50 local admitted
train corpora. E1407 is invalid before execution: training would report a zero
auxiliary loss, not an outcome.

E1409 is the corrected pre-execution endpoint. It trains the same detached,
masked-prefix head at every non-root binder reference, which provides label
coverage, but retains the decoder intervention only for repeated legal
references. The completed resolved lexer graph is label-only; no marker
spelling, external name, template text, or future declaration enters the head.
The fixed 354-step E1333 scratch recipe and E1410 held-out n=5 endpoint retain
decoder weight 4. The 25 named AgentEvals grader outputs are the tracked
metrics; `@agentv/core` only executes/publishes them. Success requires
strict-v2 above 0.6 without regressions versus E1381, else reject without a
scalar sweep. The checkpoint is local-only and may never sync, serve, resume,
parent, or promote.

E1409 completed exactly 354 capped CPU scratch steps with `stopped_on: steps`
and local checkpoint `outputs/runs/e1409_v296_e1333_prefix_binder_presence/checkpoints/last.pt`.
E1410 is rejected: all headline named grader metrics exactly match E1405/E1403
(parse 1.0, meaningful-program 0.8, strict-v2 0.6, contract/fidelity recall
0.8167, structure 0.54202, component recall 0.7810, reward 0.8958, two
fallbacks, zero timeouts). `@agentv/core` executed the 25 named graders with
zero errors and no aggregate metric. Do not sweep the prefix-head loss or decode
weight; this checkpoint is local-only, rejected, and never sync/promote/serve/
resume/use as parent.

E1411 is preregistered as a decoder-only control on the completed E1409
checkpoint. E1410 closes the Form field array with `:slot_4` still absent,
while E1399 reaches that slot but repeats the Input binder. E1411 enables only
the existing deterministic nested-array completion rule: it keeps a legal Form
field array open while an opaque request slot remains missing and another legal
path exists. It uses parser state and opaque contract ordinals only. The fixed
held-out n=5 endpoint must exceed strict-v2 0.6 with no named-metric regression
versus E1381, else reject without a decoder sweep.

E1411 is rejected. Required nested-array completion regresses held-out n=5 to
parse 0.8, meaningful-program 0.6, strict-v2 0.4, contract/fidelity recall
0.6167, structure 0.39298, component recall 0.5810, reward 0.7060, and one
timeout. `@agentv/core` executed the 25 named graders with zero errors and no
aggregate metric. Do not tune or reuse this deterministic array-completion path.

E1412 is preregistered as a fresh E1333 scratch interaction arm. E1399's
component-edge head provides the missing Form capacity but repeats forward
`b7`; E1410 avoids that alias but closes the Form array before `:slot_4`.
E1412 trains exactly those two isolated objectives together and enables only
the fixed component-edge weight 1 and prefix-reference weight 4 at decode.
Both heads use grammar-token/opaque-ordinal structure only. The held-out n=5
endpoint is fixed: strict-v2 must exceed E1399's 0.6 without regressing its
parse, meaningful-program, contract/fidelity recall, component recall, or
timeout metrics. `@agentv/core` will execute/publish the 25 named graders;
it is not a metric or aggregate. This scratch checkpoint is local-only and
cannot be synced, served, resumed, used as parent, or promoted.

E1416 completed at 354 CPU scratch steps (last loss 16.7465; SHA
`33580eceba2fde657f81d19d0a3d75f9b0405356431a8b2ff43d6fb77e784a6f`).
E1417 is invalid for the intended current-version comparison: post-run recipe
audit shows E1398 used `component_edge_alignment_loss_weight=1`, while E1416
incorrectly used 0. The observed named metrics remain durable evidence, but do
not establish version drift or detached-head interference. E1418 must restore
that fixed alignment objective before any causal comparison.

E1418 is preregistered as that exact matched replay: component-edge loss 1 and
component-edge alignment loss 1, with no added auxiliary head. E1419 fixes the
held-out n=5 endpoint. The 25 named AgentEvals graders are the tracked metrics;
`@agentv/core` only executes/publishes them. The resulting checkpoint is
local-only and never sync/promote/serve/resume/use as parent.

E1418 completed the locked 354-step CPU scratch chain with `stopped_on: steps`,
last loss 19.4225, and SHA
`63da6830008ec320417dfbaeb6822dabd35546a772f4a96472b3588f2fb0df5f`.
E1419 exactly matches E1399 on every preregistered headline metric: parse and
meaningful-program 1.0, strict-v2 0.6, contract precision/recall and fidelity
1.0, structural similarity 0.6513, component recall 0.9429, reward 0.9616,
one fallback, and zero timeouts. The 25 named AgentEvals graders executed with
zero errors through `@agentv/core`; the SDK is execution/publishing metadata,
not a metric or aggregate. This establishes current v296 base compatibility
and makes E1416/E1417's mismatch an omitted alignment-loss recipe error, not a
version or detached-head regression. The control is complete: do not sweep it,
sync it, serve it, resume it, use it as a parent, or promote it.

E1420 is a completed support audit, not a model result: the immutable E1333
mixture has 471 admitted records and 1,011 forward binder references, but zero
repeated forward references before their declarations. The two E1419 strict-v2
failures therefore target a structural state absent from the training data.
The next intervention must extend the canonical data producer with paired,
schema-valid examples of intentional sharing and distinct-slot conflicts, then
pass the strict build quality report, rejection ledger, feedback review, and a
new preregistered scratch arm. It must not add a global reuse ban, tune an
inactive head, or change any evaluation gate.

E1421 is preregistered to build the smallest strict corpus extension that
represents the missing decision. The canonical typed ProgramSpec generator will
emit 48 variants over its existing viewport, state, and content axes, paired as
root-level distinct slot-bearing `Input` references and intentional repeated
slot-free `Input` sharing. It composes those roots with the immutable E1333
records under the existing strict verifier, deduplication, exposure, and
held-out n-gram-decontamination policy. No prompt projection, model, decoder,
or gate changes are allowed. It is trainable only if strict admission retains
at least one raw ProgramSpec from each arm with zero verifier, quality, and
n-gram-decontamination rejection; the quality report, rejected ledger, and
synthesis feedback determine whether a single new scratch arm is preregistered.

E1421 is rejected by that unchanged quality loop: its published strict build
retains 665 records but rejects 210 candidates for `too_few_placeholders`; the
slot-free sharing arm cannot independently meet the content floor. There are no
verifier or n-gram-decontamination rejections. The feedback names the
ProgramSpec producer as the repair target, so E1422 changes only that producer:
the shared `Input` stays slot-free while two separate `TextContent` binders
carry the required opaque slots. E1422 may reach training only if both direct
paired arms survive strict admission; the failed E1421 corpus is never trained.

E1422 clears that admission check: its immutable v45 snapshot retains 704
records, has zero verifier and n-gram-decontamination rejection, and retains
both direct paired arms. Its 102 remaining quality-floor rejections and 11
reserved-structure drops are recorded as broader synthesis feedback, not
reasons to weaken a gate. E1423 is therefore the single fresh no-parent CPU
scratch control: it changes only E1418's corpus to E1422 while preserving the
entire supported loss stack. It may be evaluated only after `stopped_on: steps`
and remains local-only under either outcome.

E1423 is invalid: after correcting the CLI spelling to `--lr` and making the
control's required lexer/tree compiler mode explicit, the canonical 110-second
cap interrupted the run at step 6 before its first 15-step full-state
checkpoint. Those partial metrics are not evidence and are neither evaluated
nor resumed. E1424 restarts the same fresh E1422 comparison with the sole
operational change of a full-state checkpoint every step, enabling a
single-writer continuation chain within the cap.

E1424 completed its 354-step fresh E1422 chain with `stopped_on: steps`, last
loss `21.2268`, local-only checkpoint SHA
`d2543766cb63cc1e1585f5652065640be2f327f630c1ede3eab4004c64d9c38f`, and
no checkpoint sync. Its preregistered held-out `n=5` evaluation (E1425) is a
clear rejection: parse and meaningful-program rates are both `0.2`, strict
binder-aware meaningful rate is `0.2`, contract precision/recall and
placeholder fidelity are `0.2`, structural similarity is `0.154`, component
recall `0.2`, reward `0.197`, with four decode timeouts and four parse errors.
The pinned `@agentv/core` SDK executed all 25 named AgentEvals graders with
zero execution errors; it is runner/publisher metadata only, not a metric or
aggregate. E1424/E1425 is local diagnostic evidence only and must not be
promoted, served, synced, or reused as a parent.

E1426 is the controlled follow-up admission check. E1422's 704-record rebuild
did not isolate the 48 paired records from E1333's 471 rows: it also changed
source proportions and admitted language-contract material. E1426 therefore
retains E1333 as immutable roots, adds only the verified paired ProgramSpecs,
and disables every auxiliary producer and derivative. It reaches training only
if every E1333 id survives unchanged and both paired patterns survive strict
admission.

E1426 is rejected before training: its generic `existing+programspec` build
retained 470 records, dropped 13 E1333 ids, and did not retain paired-pattern
provenance in admitted records. The strict verifier rejected zero rows, but
the 17 quality drops and re-filtered baseline make this an invalid causal
comparison. The next repair belongs in the data builder: an explicit immutable
baseline append path, not weaker quality gates or another train on E1426.

E1427 uses the corresponding explicit immutable-baseline builder mode. It
preserves every E1333 row while applying the unchanged strict admission stack
only to the paired additions; training remains conditional on exact baseline
preservation and two-pattern survival.

E1427 is rejected before training: its first marker-bypass implementation
still allowed later semantic deduplication to remove 13 E1333 roots. E1428
then restored all 471 baseline rows byte-for-byte, but is also rejected: its
mixed ProgramSpec input admitted 12 rows, six without either declared
forward-reference fact. Its local E1429 train stopped at step 37 on the wall
time budget; it is both partial and confounded, so it has no evaluation.

E1428 repeats this admission check under `harness.train_data` v46, which
restores original derived rows after candidate-only gates to enforce the
immutable-baseline contract at the serialized-record boundary.

E1430 is the corrected admission check under `harness.train_data` v47. It
selects ProgramSpecs by their declared `forward_reference_pattern` facts,
requires both `root_distinct_slot_inputs` and `root_shared_slot_free_input`,
and carries those facts into each admitted record's provenance. It can unlock
one fresh local-only comparison only if every E1333 row remains byte-identical
and every supplement row has one of those two facts.

E1430 is invalid before admission: although fact filtering retained three
roots for each declared pattern, the executed command omitted
`--preserve-derived-records`, contrary to its locked recipe, and 11 baseline
rows were missing. E1431 repeats the same v47 build with that flag explicitly
bound; no training is eligible until its serialized snapshot passes the exact
baseline and fact-provenance checks.

E1431 passes: it contains 477 rows with all 471 E1333 rows canonically
byte-identical, exactly six fact-provenanced additions (three
`root_distinct_slot_inputs`, three `root_shared_slot_free_input`), zero quality
or verifier rejections, and no feedback recommendations. E1432 completed its
sole fresh local-only 354-step v296 comparison on that corpus with
`stopped_on: steps`, last loss `15.4744`, and local checkpoint SHA
`d4647d64f1ab70b49bce5b8abcac4489320e140281ed34a04a7a7b4ec522c941`.
Checkpoint sync remains disabled. E1433 is preregistered for the held-out
`n=5` strict compiler-tree evaluation. E1433 rejects the exact six-record
addition: four of five rows timed out into parse errors, and the defined grader
metrics are parse/meaningful/strict-v2/contract precision/contract
recall/fidelity/structure/component recall `0.2`, reward `0.1874`, and exact
match `0.0`; AST node and edge F1 are `1.0` on their five defined samples.
The 25 named AgentEvals grader outputs are recorded with their defined sample
counts in the E1433 JSON record: six graders are undefined for this diagnostic
subset (`canonical_exact`, `language_validity`, `ref_graph_exact`,
`target_composite`, `target_correctness`, and `target_efficiency`) and remain
`null`, not zero. `@agentv/core` executed/published the grader bundle with
zero errors, but it is runner/publisher metadata only. This is a held-out
`n=5` scratch diagnostic, not a full ship-gate result. E1432/E1433 are
rejected and never eligible to sync, serve, parent, or promote.

E1434 is preregistered to isolate the missing exposure rather than tune a
decoder: the same six fact-provenanced rows receive an explicit
`paired_forward_reference` source family, while the 471 E1333 rows must remain
byte-identical. A locked mixture weight of `0.25` gives that six-record family
materially more exposure than the rejected uniform arm, without relabeling
unrelated ProgramSpecs. The arm remains local-only and must pass the same
strict build-quality checks before its fresh 354-step train can begin.

E1434 admission passes: the 477-row corpus has zero baseline canonical JSON
mismatches, six `paired_forward_reference` rows split three per declared fact,
and zero quality or verifier rejections. The feedback report records a `0.7391`
raw duplicate share for the paired producer; its emitted candidate is to reduce
expansion or diversify templates in a later producer experiment. This run keeps
the admitted six rows and unchanged strict filters, and tests exposure only.
E1434 completed its 354-step local scratch train with terminal checkpoint SHA
`2241a4d4a159471aea423a96fb5e5c4fc34f4b8d8245b612bfa857243113bcb8` and
step-354 loss `11.3735`; a final no-op resume overwrote the summary's loss with
`0.0`, so `metrics.jsonl` remains the authoritative final optimizer-loss
source. E1435 is preregistered for the same held-out `n=5` grader vector.
E1435 fails every defined primary grader: all five rows reach parse-error
timeouts, so parse, meaningfulness, strict-v2, contract, fidelity, structure,
component recall, and reward are `0.0`. AST F1 is undefined (`null`, `n=0`),
not zero. The exposure arm also changed the sampler from E1432's uniform path,
so E1436 is preregistered as the required same-sampler E1333 control before
assigning the regression to the six paired examples.

E1437 rejects that sampler path: the same family-weighted sampler on immutable
E1333 collapses parse to `0.2`, strict-v2 to `0.0`, and produces four decode
timeouts. The tracked grader vector is recorded in JSON (AST edge F1 `0.6`,
node F1 `0.8333`, contract/fidelity/component recall `0.2`, structure `0.128`,
reward `0.1922`). Thus E1435 is not evidence that the paired examples alone
caused the failure; no mixture-weight sweep follows.

E1442 targets the rare Tabs/Card structural family with generic verified
train-only ProgramSpecs. It explicitly excludes held-out programs, prompts,
labels, and opaque-slot identities; strict admission and decontamination remain
unchanged.

E1442 admission passes under that unchanged strict policy: 48 generic
ProgramSpecs produce 24 admitted Silver records, all containing Tabs, TabItem,
Card, and TextContent. Eight candidates are removed by deduplication and 16 by
verification quarantine; there are zero decontamination, parse, quality, or
independent-judge failures among the admitted rows. The producer yield is
`0.5`, with no emitted feedback recommendation. This is a documented
data-production result, not a reason to relax the filter.

E1443 is preregistered as the matched corpus comparison: retain E1440's
uniform E1333 primary stream, corrected cumulative replay accounting, 354-step
CPU scratch recipe, seed, architecture, losses, decoder, and `0.125` replay
fraction, but replace the six paired forward-reference records with the E1442
Tabs/Card corpus. It must not use the rejected family-weighted sampler. E1444
locks the held-out `n=5` strict compiler-tree endpoint. Its scoreboard will
list all 25 named AgentEvals grader metrics and their defined sample counts;
`@agentv/core` remains runner/publisher metadata only. Neither local scratch
arm is eligible for promotion, serving, checkpoint sync, or use as a parent.

E1443 completes the locked 354-step CPU scratch train with `stopped_on: steps`,
last loss `21.2351`, 310 primary and 44 replay examples, and effective replay
fraction `0.1242938`. The terminal summary and full-state checkpoint agree on
the cumulative replay counters. E1444's strict held-out `n=5` evaluation then
records all 25 named AgentEvals graders: parse, meaningful-program,
contract precision/recall, placeholder fidelity/validity, and raw syntax are
`1.0`; strict meaningful-v2 is `0.6`; component recall `0.942857`; reward
`0.9616`; structural/tree-edit similarity `0.6385`; AST node/edge F1
`0.745116`/`0.569968`; exact match `0.0`. The six inapplicable graders remain
`null` with defined count zero, rather than being treated as failures. There
are two fallbacks and zero decode timeouts. The complete named vector and each
defined count are stored in the JSON ledger. `@agentv/core` ran and published
the bundle; its benchmark pass rate is not a metric or decision input. This
recovers the E1419 validity/fidelity/reward envelope and improves E1441, but
the local held-out `n=5` scratch result remains non-promotable, non-servable,
and cannot be used as a parent.

E1438 is the next isolated design: build the same six accepted pairs as their
own strict corpus, then use the existing replay sampler against a uniform E1333
primary stream. This changes exposure without reintroducing the rejected
family-weighted sampler.

E1439 completed its optimizer steps but is invalid for evaluation: its terminal
summary reports zero effective replay and zero replay examples, and its durable
full-state checkpoint lacks cumulative replay counters. Segment-local counters
cannot prove the requested exposure across cap/resume boundaries. A fresh replay
arm requires resume-safe cumulative accounting first.

E1412 completed its locked 354-step CPU scratch chain with `stopped_on: steps`,
last loss 17.0087, and local checkpoint SHA
`78a2e601d376e25a5f924a3fff26c226e8bad19877046a9d57cc0eb690fd73f3`.
Checkpoint sync remains disabled; it is eligible only for the preregistered
E1413 held-out evaluation.

E1413 rejects the interaction: held-out n=5 gives parse 1.0,
meaningful-program 0.8, strict-v2 0.4, contract/fidelity recall 0.75,
structure 0.47854, component recall 0.7429, reward 0.8758, two fallbacks, and
zero timeouts. The Form prediction still reuses forward `b7`. `@agentv/core`
executed the 25 named graders with zero errors and no aggregate metric. Do not
sweep either component-edge or prefix-reference weight; E1412 is rejected,
local-only, and never sync/promote/serve/resume/use as parent.

E1414 is preregistered as a fresh E1333 interaction arm. It retains E1398's
component-edge capacity objective but replaces E1412's rejected binary
prefix-reference head with E1402's exact opaque binder-to-slot ownership head.
The ownership target is evaluated only at repeated legal binder paths and uses
opaque ordinals, grammar IDs, and prompt context, never marker spellings,
external names, template text, or future declaration tokens. E1415 fixes the
held-out n=5 endpoint: strict-v2 must exceed E1399's 0.6 with no regression in
parse, meaningful-program, contract/fidelity recall, component recall, or
timeout count. `@agentv/core` will execute/publish the 25 named graders; it is
not a metric or aggregate. The checkpoint is local-only and cannot be synced,
served, resumed, used as parent, or promoted.

E1414 completed its locked 354-step CPU scratch chain with `stopped_on: steps`,
last loss 16.7857, and local checkpoint SHA
`e8e2556dd93d7c7c6c2f94333f3eab2e9f40f426a8f820c9b0614cb94e2b721a`.
Checkpoint sync remains disabled; it is eligible only for E1415.

E1415 rejects the exact-ownership interaction. Its prediction and all headline
named metrics exactly match E1413: parse 1.0, meaningful-program 0.8,
strict-v2 0.4, contract/fidelity recall 0.75, structure 0.47854, component
recall 0.7429, reward 0.8758, two fallbacks, and zero timeouts. `@agentv/core`
executed the 25 named graders with zero errors and no aggregate metric. Exact
ownership is inactive in this capacity interaction; do not sweep either weight.
E1414 is local-only and never sync/promote/serve/resume/use as parent.

E1416 is preregistered as the matched v296 component-edge-only replay. E1399's
v291 component-edge arm is strong on capacity, while E1413/E1415 are identical
v296 joint regressions despite detached added heads. E1416 therefore removes
both added heads and holds the E1398 recipe and E1417 held-out n=5 decode
endpoint fixed. The 25 named AgentEvals graders are the tracked metrics;
`@agentv/core` only executes/publishes them. This checkpoint is local-only and
cannot be synced, served, resumed, used as parent, or promoted.

E1409's first capped CPU window reaches 36/354 steps with a local full-state
checkpoint and a nonzero prefix-reference objective. It stops on the canonical
wall-time budget and is incomplete, unevaluable evidence; resume only its own
full-state chain to the locked 354-step endpoint.

E1360 completed its locked 354-step CPU scratch chain (local SHA
`a41d44dbdafad57daa4e0923d48e853b95b07f517e5f3845bf75c3fed855f465`).
E1361's strict one-row Form evaluation exactly matches E1356/E1359's 25 named
grader outputs and `root = TextContent(":slot_0")`: parse 1.0, strict-v2 0,
fidelity/contract recall 0.1667, structure 0.1148, component recall 0.1429,
reward 0.657, no timeout, and two fallbacks. `@agentv/core` executed the
graders with zero errors and no aggregate metric. The margin restores five of
eight semantic-plan choice changes, but does not repair b7. Reject E1360; it
remains local-only, unsynced, unpromoted, unserved, unresumed, and non-parentable.

E1362 is a fresh no-parent E1333 small-architecture control. The corpus audit
already finds the necessary multi-binding Form ordering, so it retains
semantic-exhaustive compiler alignment and changes only teacher-forced LTR CE
weight from 0.5 to 1.0 to reinforce late declaration realization. Strict policy,
named graders, grammar authority, and fallback behavior remain locked.

E1362 completed its locked 354-step CPU scratch chain (local SHA
`7bfbc9322fe631e15558bd086eea4237c1a1a368145f8e0200d3d635cc9dd82d`).
E1363's strict one-row Form evaluation exactly matches E1356/E1359/E1361's 25
named grader outputs and `root = TextContent(":slot_0")`: parse 1.0, strict-v2
0, fidelity/contract recall 0.1667, structure 0.1148, component recall 0.1429,
reward 0.657, no timeout, and two fallbacks. `@agentv/core` executed the
graders with zero errors and no aggregate metric. The stronger LTR CE retains
three semantic-plan choice changes but does not repair the b7 dead end. Reject
E1362; it remains local-only, unsynced, unpromoted, unserved, unresumed, and
non-parentable.

E1364 is preregistered as a fresh no-parent E1333 small-architecture control.
E1363 reaches the `b6 =` declaration before its completion forest empties, and
the existing binder-specific component planner has a declaration-path decode
bias that was never activated in the preceding arms. E1364 retains E1358's
semantic-exhaustive supervision and changes only
`binder_component_plan_decode_weight` from 0 to 2 under the required compiler
tree mode. Strict policy, named
graders, grammar authority, and fallback behavior remain locked.

E1364 completed its locked 354-step CPU scratch chain (local SHA
`e452eee2f1cac2836903b39e65fb6998767cf8c25416d61e2d172d926096a364`).
E1365's strict one-row Form evaluation exactly matches E1356/E1359/E1361/E1363's
25 named grader outputs and `root = TextContent(":slot_0")`: parse 1.0,
strict-v2 0, fidelity/contract recall 0.1667, structure 0.1148, component
recall 0.1429, reward 0.657, no timeout, and two fallbacks. `@agentv/core`
executed the graders with zero errors and no aggregate metric. The binder
planner applied twice but changed no choice; the completion forest still emptied
before b7. Reject E1364; it remains local-only, unsynced, unpromoted, unserved,
unresumed, and non-parentable.
