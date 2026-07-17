# E252 prerequisite: exact-state semantic counterfactuals (2026-07-16)

Status: **measured negative; no corpus and no training**.

E249 proved that exact compiler constraint shadows encode lexical legality but
not semantic preference. This follow-up adds a fail-closed path that replays
multiple grammar-legal tokens from the same production compiler canvas, grades
each completed output with the independent data judge and meaningful-program
verifier, and labels only the verified Pareto frontier against failed or
strictly dominated legal alternatives. No prompt-, component-, or output-string
special cases participate in candidate selection or labeling.

## Admission contract

- Candidate tokens come from the production compiler tree's
  `allowed_id_set` at one exact `pre_canvas` and position.
- Every continuation reuses the same context, canvas, position, seed, decode
  policy, checkpoint, tokenizer, and slot contract.
- A good token must pass both `independent_judge` and
  `_is_meaningful_program`; quality uses placeholder fidelity, component
  recall, structural similarity, and reward.
- A persisted `counterfactual_decision` is accepted only when it matches a
  qualified `counterfactual_probe` and its Pareto labels recompute exactly.
- Local preference training now rejects `constraint_shadow` events. Those
  events remain deterministic decoder-legality evidence only.

## Bounded diagnostic

The completed diagnostic used the E228 parent, CPU, seed 250, one accepted E230
source record (`lc_forward_reference`), one grammar state, and two legal
continuations. Trace identity is
`f04946404c984c36b524274a221d1e99`; checkpoint SHA is
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`.
An earlier 2-record/3-state/4-candidate pilot was stopped before trace emission
because its CPU inner loop was too large; a one-state/two-candidate sizing run
then completed before probe persistence was added.

| Measure | Result |
| --- | ---: |
| Production traces accepted | 0 / 1 |
| Exact states replayed | 1 |
| Grammar-legal candidates | 2 |
| Independent-judge pass | 2 / 2 |
| Meaningful-program pass | 0 / 2 |
| Verified candidates | 0 / 2 |
| Qualified preference events | 0 |

At position 1, legal tokens were `<BIND_0>` and `NL`. Both raw continuations
were invalid, both production-finalized to the same
`root = Button(":cta.label")` fallback, and both failed meaningful-program
verification with `low_component_recall:0.00`. Placeholder fidelity,
component recall, and reward were all zero; structural similarity was 0.06.
The independent judge alone passed both outputs, demonstrating why it cannot be
the sole admission gate for completed-layout preference data.

## Decision

Do not create an immutable counterfactual corpus and do not run E252 training
from this probe. The next hypothesis must first produce distinct, complete
same-state continuations without collapsing through generic finalization. The
decoder-legality tree remains deterministic; this result concerns semantic
preference evidence, not whether a partial lexical state is parseable as a full
OpenUI document.

Machine-readable evidence:
[`quality-matrix-v10-e252-prerequisite-results.json`](quality-matrix-v10-e252-prerequisite-results.json).
