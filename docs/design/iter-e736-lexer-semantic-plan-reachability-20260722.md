# E736 lexer semantic-plan reachability

**Date:** 2026-07-22
**Decision:** retain the fail-closed harness fixes; reject the decode recipe
**Evidence:** [`iter-e736-lexer-semantic-plan-reachability-20260722.json`](iter-e736-lexer-semantic-plan-reachability-20260722.json)

E735's corrected root-arity head was causal only in the sense that it removed
impossible tail predictions; its one-reference prediction already agreed with
all three smoke outputs. The remaining strict failures were missing component
families. E736 traced the existing prompt-semantic plan through the canonical
lexer compiler before spending another training cycle.

The central lever registry declared semantic-plan family scoring and its margin
choice-only, and model generation rejected them before the lexer compiler could
run. After exposing only those two genuinely dual-path levers, r1 accepted the
configuration but recorded zero applications: compiler-tree path ranking never
called the semantic-plan scorer. r2 threaded the batch row and grammar state
through that shared selector and produced five applications and five choice
changes, proving reachability. It also showed that topology-free family evidence
must not override the trained root-role head. r3 restricted obligations to
bound families, but the lexer used bare component symbols while coverage
accounting recognized only choice-codec marker prefixes, so fulfilled families
were forced repeatedly. r4 counts tokenizer component IDs for both codecs and
reduces the treatment to three applications and two choice changes without
duplicate family forcing.

All accepted evaluations reused E735's local-only checkpoint and the same three
frozen smoke records under `strict_compiler_tree`, honest slot-constrained
decode, no unconstrained fallback, an eight-second per-record timeout, and a
160-symbol canvas. Every run completed locally in under ten seconds with zero
decode timeouts and emitted AgentEvals plus an AgentV SDK bundle.

| Arm | Plan apps / changes | Parse | Meaning-v1 | Strict-v2 | Fidelity | Structure | Recall | Reward | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E735 control | 0 / 0 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 0/1 |
| Slot-owner off diagnostic | 0 / 0 | 1.0000 | 0.3333 | 0.0000 | 0.3333 | 0.1242 | 0.1667 | 0.7250 | 0/1 |
| r1 accepted but inert | 0 / 0 | 1.0000 | 0.6667 | 0.0000 | 0.5278 | 0.5614 | 0.4167 | 0.8073 | 0/1 |
| r2 root + bound reachability | 5 / 5 | 1.0000 | 0.6667 | 0.0000 | 0.4444 | 0.2130 | 0.3333 | 0.7403 | 0/1 |
| r3 bound-only ownership | 5 / 4 | 1.0000 | 0.6667 | 0.0000 | 0.4444 | 0.4586 | 0.3333 | 0.7683 | 0/1 |
| r4 codec-neutral coverage | 3 / 2 | 1.0000 | 0.6667 | 0.0000 | 0.4444 | 0.4586 | 0.3333 | 0.7683 | 0/1 |

r4 recovers `Callout` on the callout prompt and preserves the button result, but
the hero's selected `Card` closes empty and falls back to a generic symbol-only
button. Aggregate fidelity, structure, recall, and reward remain below the E735
control, and strict-v2 remains zero. Keep the generalized reachability,
root-ownership, and codec-neutral coverage fixes at their default-off settings;
reject semantic-plan weights 4/2 for this checkpoint. No training ran, no
checkpoint was created, and nothing was synced, promoted, or served.

Two support failures are excluded from evidence: the first eval preflight
resolved the shared checkout's older CLI and rejected the atomic policy flag,
and a later read-only summary loop had invalid shell-wrapper syntax. Neither
created an experiment artifact.
