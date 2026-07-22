# E740 — lexer semantic-role family scoring

**Date:** 2026-07-22
**Decision:** retain reachability fixes; reject weight and checkpoint promotion
**Evidence:** [`iter-e740-lexer-semantic-role-family-20260722.json`](iter-e740-lexer-semantic-role-family-20260722.json)

E740 traced the remaining missing-family failure through the shared
slot-to-component scorer. The lever catalog already declared
`semantic_role_decode_weight` and the learned slot-component head dual-path,
but compiler calls omitted declared role candidates and lexer forests label
inline families `component` rather than the choice codec's `component_bound`.
One row adapter now supplies the candidates to every call site, and v206 treats
both grammar-role names as the same bound component decision. No new lever or
free-form output channel was added.

Both accepted arms reuse the unchanged local E735 checkpoint and three frozen
smoke records. They ran locally on CPU with `strict_compiler_tree`, visible
declared semantic roles, schema-role weight 2, semantic-plan weight 4 / margin
2, a 160-symbol canvas, an eight-second per-record guard, and the two-minute
command cap. The only matched change is semantic-role family weight 0 versus 2.

| Arm | Slot-family apps / changes | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v206 role 0 | 8 / 3 | 1.0000 | 0.6667 | 0.0000 | 0.9167 | 0.9500 | 0.4886 | 0.4167 | 0.9120 | 2150 / 2581 ms | 0/1 |
| v206 role 2 | 8 / 3 | 1.0000 | 0.6667 | 0.0000 | 0.9167 | 0.9500 | 0.4886 | 0.4167 | 0.9120 | 2199 / 2313 ms | 0/1 |

The treatment changes the button record's first child from `TextContent` to the
declared `Button` family and removes `duplicate_subtree_spam`. It does not alter
any aggregate metric because decoding continues after consuming the only
declared template symbol, producing repeated-symbol children and retaining
`placeholder_spam`. Strict-v2 and AgentV remain zero.

Retain the shared reachability corrections because enabled, trained levers must
not silently no-op. Reject semantic-role weight 2 for promotion on this recipe;
no checkpoint was created, synced, or promoted. The next intervention should
extend the existing coverage-close behavior to lexer containers so legal
decoding stops after all declared template symbols are consumed.

The preliminary v205 r1 pair is preserved as negative evidence: both arms were
byte-identical and reported zero slot-family applications, exposing the lexer
`component` versus choice `component_bound` kind mismatch.

Post-run remediation closes the harness/configuration footgun rather than
filtering it from presentation. Canonical lever registry v12 now owns required
slot-contract and visible semantic-role companion settings. Model construction
rejects an enabled lever with missing companions before a run directory or
evaluation artifact can be created, and the retained generation-time guard
reads the same canonical slot-contract lever set. Regression coverage proves
the pre-artifact failure and the real lexer compiler family decision. The v205
runs remain labeled negative diagnostic evidence; they are not silently hidden
or treated as valid comparison arms.
