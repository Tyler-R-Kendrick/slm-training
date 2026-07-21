# E709 — final schema-value margin

Date: 2026-07-21  
Status: completed matched five-suite retained scratch improvement; not ship

E709 closes the semantic failure isolated by E708. Schema enum-slot protection
now runs after semantic and coverage biases and floors any visible-slot candidate
below the best legal non-slot value by the configured margin. Once every slot and
the carrier reference are present, a fully covered terminal `Stack` similarly
floors closure above duplicate bare placeholders. Nested closure behavior is
unchanged.

The first Rico diagnostic removed `TextContent.size` misuse but exposed a trailing
duplicate placeholder in the root list. The terminal covered-close margin removes
that duplicate. After rebasing onto main's v181 opaque-symbol work, the evaluator
first failed before scoring because legacy evaluation requests did not declare the
new typed runtime-symbol authority; that failed r4 attempt is not evidence. The
canonical evaluator now explicitly declares the historical, prompt-visible slot
suffix roles while leaving opaque production requests caller-authoritative.

The matched v181 control and v182 treatment both replay all five suites (`n=19`)
with the same v37 evaluator. Every non-Rico quality metric is identical. E709 alone
raises Rico binding-aware strict meaningfulness 0.0→1.0; Rico contract recall,
fidelity, validity, and reward remain 1.0, structure remains 0.7915, node F1
0.8419, and edge F1 0.7727. All three Rico records have empty semantic reason-code
lists.

Retain v182 with schema-value weight 5. This remains scratch-matrix evidence,
not a ship evaluation: AgentV is 0/5 and Rico's p95 target length is 190 against
the 160-token evaluation canvas. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e709-final-schema-value-margin-20260721.json).
