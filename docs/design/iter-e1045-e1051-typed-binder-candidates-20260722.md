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
