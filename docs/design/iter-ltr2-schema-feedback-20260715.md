# Iteration: schema conditioning with LTR2 feedback (2026-07-15)

This scratch experiment tested `schema_in_context=true` with the selected
seed-0 LTR2 recipe (`ltr_loss_weight=2.0`) for 64 steps. It used the unchanged
585-record remediated corpus, interval held-out loss feedback every 32 steps,
CPU execution, and no design-document context.

## Result

Held-out weighted NLL improved to **7.215** at step 64. The bounded one-record
smoke feedback probe did not parse: parse rate and reward were **0**. Its
structural similarity was **0.350**, placeholder validity **0.1333**, and
component recall **0.500**.

The matched LTR2 control reached structural similarity **0.5375** and
placeholder validity **0.200** at the selected step-64 checkpoint, with the
same zero parse and reward rates. The lower loss therefore did not translate
into better executable OpenUI output.

## Decision

Reject schema conditioning for this branch. Do not promote its checkpoint or
claim a syntax improvement. The next syntax-focused iteration should target
the malformed token sequence / serialization contract directly, while keeping
the LTR2 checkpoint and bounded AgentV smoke probe as the control.

The complete train summary, scoreboard, AgentEvals JSONL, and AgentV bundle are
listed in the companion JSON record.
