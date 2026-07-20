# E603 — final-choice and reachability trace

Date: 2026-07-20
Status: diagnostic success; not promotable or ship

E603 extends the bounded E602 score trace through the actual final legal-token
selection. It records the final token, whether a downstream score layer changed
the plan-stage winner, and each traced candidate's aggregate post-plan score
delta. The instrumentation changes no legality or ranking policy.

The first capped OOD `n=4` run omitted the matched visible-contract overrides.
It completed normally but is not comparable to E602: structure fell to 0.2175,
meaningful-v1 and reward to 0, and AgentV remained 0/1. The mismatch is
preserved as `e603-e602-final-choice-trace-r1`, not used as causal evidence.

The canonical `e603-e602-final-choice-trace-r2` retry restores the complete
E602 policy and reproduces every headline metric: syntax 1.0, meaningful-v1
0.5, strict meaning-v2 0, fidelity 0.5917, validity 0.7550, structure 0.5169,
component recall 0.6250, reward 0.8115, AST-node F1 0.5754, and AST-edge F1
0.4143. AgentV is 0/1.

The downstream-score-overwrite hypothesis is false. All four plan-stage
winners remain the actual final legal token, and every traced post-plan delta
is zero:

- dashboard: `Button` remains final but binding zero is never referenced;
- gallery: `ImageGallery` remains final but binding zero is never referenced;
- modal: `Button` remains final and is referenced by the later `Stack`;
- auth: `Input` remains final and is referenced by the later `Stack`.

Structural choice decoding makes the last expression the root. The official
canonicalization path removes unreachable earlier bindings, so dashboard's
seeded `Button` and gallery's seeded `ImageGallery` disappear from the rendered
program. Also, E602's “before” token was the pre-plan winner, not a matched
seed-zero control: the regular plan weight already selects the modal/auth
families and can select dashboard `Button`. The actionable boundary is
reachability and missing-family coverage, not first-choice ranking or an
attempt selector.

No checkpoint was created, promoted, or synced. Next test a bounded,
prompt-derived missing-family coverage intervention and require planned
bindings to remain reachable before considering promotion.

Evidence: [JSON](iter-e603-final-choice-reachability-trace-20260720.json).
