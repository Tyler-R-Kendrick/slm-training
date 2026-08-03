# Autotrain c1828: wide compiler draft null

**Verdict:** reject `grammar_draft_window=16` as a material compiler-tree
optimization. It changes neither quality nor decode work, and its 0.97% p50
improvement is below the preregistered 5% efficiency floor while compiler time
is 0.71% worse.

| Arm | Params | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | Compiler ms | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| draft 8 control | 1,608,962 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 60 | 12 | 6756.51 | 2959.37 |
| draft 16 | 1,608,962 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 60 | 12 | 6804.31 | 2930.73 |

Both arms use 8,960 compiler-prefill tokens, 3,072 canvas tokens, and build
92,013 completion states. The candidate effective eval config records draft
window 16 versus 8, but the compiler emits the same spans and performs the same
neural work. The latency delta is therefore timing noise, not a causal
efficiency win.

Training telemetry provides the stronger successor signal: 42 draws contain
only 31 unique records and 23.84 effective records, with one record repeated
four times. The next arm retains the all-family legal margin recipe and changes
only the canonical online mixture policy from `with_replacement` to
`capacity_aware`. It tests whether reduced repeat concentration improves
effective exposure and guarded OpenUI quality without adding parameters,
changing the corpus, or weakening constrained decoding.

This remains fixture evidence: `n=3<20`, MPR and recall miss their gates, AST
and canonical equality are zero, and held-out, adversarial, OOD, and full Rico
were not run. Both arms use seed 101828, 21 CPU scratch steps, batch size 2, and
1,608,962 trainable parameters. The checkpoints are byte-identical because the
treatment is eval-only, SHA `e66960b1...30bb6c`; they are local no-sync
artifacts and are never reusable, promotable, syncable, or shippable. Lean is
`not_applicable:screening`.

Machine evidence:
[`autotrain-cycle-1828-wide-draft-null.json`](autotrain-cycle-1828-wide-draft-null.json).
