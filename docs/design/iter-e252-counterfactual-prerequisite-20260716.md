# E252 prerequisite: exact-state semantic counterfactuals (2026-07-16)

Status: **compiler environment repaired; evidence still insufficient; no corpus
and no training**.

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

## Invalidated diagnostic and generalized repair

The original trace `f04946404c984c36b524274a221d1e99` reported zero
admissible events, but that result is invalid as model-quality evidence. The
isolated worktree lacked the OpenUI bridge's Node dependencies, and three
supposedly in-process paths silently depended on that bridge: compiler schema
loading, AST-completion validation, and quality parsing. The candidates then
collapsed through generic finalization. This was an environment-dependent
harness result, not evidence that the checkpoint could not produce semantic
counterfactuals.

The repair is contract-derived rather than example-derived:

- commit the pinned official schema with a live parity checker and use it when
  the bridge is unavailable;
- use the Lark AST backend for offline completion and hybrid parsing for judge
  and grammar scores;
- require grammar-level newline separators between statements;
- preserve trailing newlines while decoding partial prefixes; and
- intersect DSL-native candidate admission with the complete compiler forest,
  replacing parser-error-string checks.

The grammar change updates the language contract from `f2d0c69ba5849ef9` to
`e3bf2f98f043e9a8`.

## Corrected bounded diagnostic

The canonical corrected probe used the unchanged E228 parent, CPU, seed 253,
the first 32 E230 records, two states per record, and four candidates per state.
Trace identity is `881de9b36dc1d4c5f4242d2efadc46a1`; checkpoint SHA is
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`.

| Measure | Result |
| --- | ---: |
| Production traces accepted | 32 / 32 |
| Exact states replayed | 64 |
| Grammar-legal candidates | 192 |
| Independent-judge pass | 92 / 192 |
| Verified candidates | 10 / 192 |
| Qualified preference events | 6 |
| Qualified prompt groups | 3 |
| Set-valued events | 3 |
| Train / held-out events | 6 / 0 |

The six events come from `lc_forward_reference`,
`render_sample_edit_268f71ed9621`, and `rico_train_15`. Their persisted judge
probes include distinct valid continuations and three set-valued comparisons,
so the repaired path now measures semantic alternatives rather than compiler
legality shadows. The stable group split assigns all three groups to train,
however, leaving no independent held-out recurrence evidence.

## Decision

Do not publish a training corpus and do not run E252 training from this probe.
The path is now technically valid, but six events from three train-only groups
are too small and cannot measure held-out recurrence. Expand evidence collection
under the same immutable identity until at least one qualified held-out group
exists; then publish `events.jsonl`, the full qualified judge probes, and their
fingerprints together. No split may be reassigned by hand.

Machine-readable evidence:
[`quality-matrix-v10-e252-prerequisite-results.json`](quality-matrix-v10-e252-prerequisite-results.json).
