# E276 — decision-signature alignment

Date: 2026-07-17
Status: **completed; remaining conflict is sparse signature coverage**

E276 keeps E275's frozen train direction unchanged and evaluates held-out
metrics at a finer semantic granularity. A decision signature is a stable hash
of `decision_kind`, legal token IDs, good token IDs, and bad token IDs. These
fields come from the grammar/AST verifier event, so the grouping generalizes
across prompts and does not match output strings or individual test cases.

The run used CPU, the unchanged E228 checkpoint, all 239 independently judged
E261 events, legal-token-conditioned probability, unit-normalized train metric
gradients, kind-level train strata, and signature-level held-out strata. The
branch was clean, rebased on current `origin/main`, and proved `0 behind / 1
ahead` at `696b7df` immediately before the read-only profile. No optimizer or
checkpoint mutation occurred.

## Result

The held-out split contains 21 semantic signatures. Nine have no exact train
counterpart. Signature-level evaluation exposes 17 regressing objectives across
seven signatures:

- four uncovered signatures: bound-child `<BIND_1>` selection, bound `Card`,
  bound `Button`, and `<SYM_4>` selection;
- three nominally covered but sparse signatures: root-child `<BIND_1>` has one
  train example, bound `TextContent` has two, and `STR:row` versus
  `STR:column` has three.

The four uncovered regressing signatures account for seven objective failures.
The three sparse covered signatures account for the other ten. Coarse
decision-kind averaging therefore concealed both missing support and
within-kind disagreement. The result explains why E275 could align every
kind-level train objective yet still oppose held-out component-bound mass and
literal loss.

## Decision

Do not train. The next lever is the synthesis pipeline: generate future train
events to cover grammar-derived decision signatures, independently judge each
prompt/output and counterfactual label before admission, and enforce minimum
per-signature support without copying held-out prompts. Coverage targets must
be derived from signature metadata rather than hard-coded token or component
names. After rebuilding the judged corpus, rerun this profile before any
optimizer command.

Machine-readable evidence:
[`quality-matrix-v10-e276-signature-alignment-results.json`](quality-matrix-v10-e276-signature-alignment-results.json).
