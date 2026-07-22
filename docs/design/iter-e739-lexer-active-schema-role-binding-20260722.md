# E739 — lexer active-schema role binding

**Date:** 2026-07-22  
**Decision:** retain the shared scorer; reject checkpoint promotion  
**Evidence:** [`iter-e739-lexer-active-schema-role-binding-20260722.json`](iter-e739-lexer-active-schema-role-binding-20260722.json)

E738 exposed a lexer-only gap: `schema_role_slot_decode_weight` was cataloged
as choice-only, and compiler-tree paths never applied the active-property slot
scorer at nested trie edges. E739 makes that existing lever dual-path. Choice
and lexer decoding now share one row adapter; lexer decisions derive authority
only from the active grammar call, public schema property, and declared
template-symbol semantic role. No free-form output channel was added.

Both accepted arms reuse the unchanged local E735 checkpoint and three frozen
smoke records. They ran locally on CPU with `strict_compiler_tree`, an honest
visible template-symbol contract, semantic-plan weight 4 / margin 2, a
160-symbol canvas, an eight-second per-record guard, and the two-minute command
cap. The only matched change is schema-role slot weight 0 versus 2.

| Arm | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v204 role 0 | 1.0000 | 0.6667 | 0.0000 | 0.7222 | 0.8333 | 0.5369 | 0.4167 | 0.8537 | 1772 / 1787 ms | 0/1 |
| v204 role 2 | 1.0000 | 0.6667 | 0.0000 | **0.8056** | **0.8833** | 0.5369 | 0.4167 | **0.8787** | 1963 / 2136 ms | 0/1 |

The targeted strict-v2 reason `placeholder_semantic_role_mismatch` falls from
one record to zero. Hero slots change from subtitle/kicker/kicker to
kicker/title/body, and callout slots change from body/heading to heading/title.
The treatment therefore removes the observed wrong-property binding instead of
filtering it from metrics.

Strict-v2 and AgentV remain zero. Repeated template-symbol reuse and missing
prompt-required component families are still present, so this bounded scratch
diagnostic is not a ship evaluation. Retain `config.levers` v11 and
`model.twotower` v204, but do not promote or sync a checkpoint.

The preliminary `e739-active-schema-role0-smoke-r1` control set
`local_files_only=true`; it is preserved but excluded from the matched pair.
The accepted r2 pair matches on `local_files_only=false`.

A v203 exact-command replay matches the v204 weight-zero control exactly, so
the new default-off call is behavior-neutral. It does not reproduce the older
E738 output even though E738's persisted policy subset appears identical. The
eval harness had manually listed only some decode weights, omitting active
fields such as component-plan and root-arity weights. `harness.model_build.eval`
v41 now derives every `*_decode_weight` from the canonical `ModelBuildConfig`
field registry and persists it beside every future scoreboard.
