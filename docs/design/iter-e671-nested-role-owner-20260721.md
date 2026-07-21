# E671 — nested role-aware array ownership

Date: 2026-07-21
Status: completed neutral scratch; rejected; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. This
was an evaluation-only scratch arm (`steps=0`, context backend `scratch`). It
emitted AgentEvals and AgentV with no timeout or fallback after all 118 compiler
tests passed.

E671 combines nested item-schema propagation, schema-allowed role targets, and
nearest-component ownership through nested arrays. All four prediction hashes
remain identical to E670, including the wrong `Form` inside Carousel. Strict v2
remains 0.7500, structure 0.7230, and node/edge F1 0.7987/0.6845. The latency
difference is not attributable on this small run.

Reject neutral v126 and restore retained E666 behavior as v127. The focused
invariant passes, but the real Carousel is model-introduced and absent from the
authored semantic plan, so the owner gate still abstains. A future retry should
use active public-schema ownership for a missing visible slot rather than
pretending the invented component was authored. No checkpoint was created,
synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e671-nested-role-owner-20260721.json).
