# E836-E838: prompt-component coverage closure

## Outcome

E836 enabled only the general prompt-to-grammar component plan on the unchanged
local E832 checkpoint. It recovered every smoke prompt-required component and
kept opaque-marker fidelity at 1.0, but all three rows still failed strict-v2
for duplicate AST-subtree spam.

E837 changed the shared decoder harness so the existing semantic-plan margin
closes an active structural array after all required grammar component counts
and opaque marker identities are present. It never reads marker text beyond
ordinal tokenizer identity and does not convert or assign marker meanings.
Smoke strict-v2 rose from 0/3 to 3/3, structure from 0.1858 to 0.6033, and p95
fell from 5.46 to 3.47 seconds. Parse, meaning-v1, and marker fidelity remained
1.0 with zero timeout or fallback.

## Held-out rejection

E838 ran the same v229 policy locally on the frozen held-out slice (`n=5`). It
did not generalize: strict-v2 was 0/5, parse 0.6, fidelity 0.5, structure
0.1389, reward 0.5508, and two records hit the 12-second per-record timeout.
The surviving failures expose schema-typed collection errors, unnecessary
binder references, and duplicate closed subtrees. AgentV remained 0/1 for each
accepted evaluation run.

## Decision

Retain the generalized, default-off coverage closure behind the already
centralized `semantic_plan_margin_decode_weight`; do not add another lever.
The broader compiler suite also exposed and removed its last shared named-marker
fixture; test records now use `:slot_0` and model-side rejection stays strict.
Reject policy promotion, checkpoint changes, sync, deployment, or ship claims.
The next arm must make still-required grammar component paths outrank unneeded
binder references before coverage completes. Template-marker conversion or
semantic marker labels are not part of that path.
