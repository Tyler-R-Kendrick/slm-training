# Autotrain c1826: bounded compiler-decision margin null

**Verdict:** reject `grammar_completion_bounds` as a compiler-tree cost
treatment. The bounded and unbounded all-family margin arms are identical on
every measured quality and decode-work counter; the bounded arm is 1.17% slower.

| Arm | Params | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | Compiler ms | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unbounded margin control | 1,608,962 | .3439 | .6667 | .3333 | 1.0 | 1.0 | .973 | 201 | 51 | 27308 | 11583.87 |
| bounded margin | 1,608,962 | .3439 | .6667 | .3333 | 1.0 | 1.0 | .973 | 201 | 51 | 27317 | 11719.02 |

Both arms also use 28,928 compiler-prefill tokens, 13,056 canvas tokens, and
70 forced row tokens without a forward. Although the candidate effective config
records `grammar_completion_bounds=true`, both `completion_bound_known` and
`completion_bound_unknown` remain zero. The strict evaluation policy routes
through compiler-tree decoding, while the bounds probe is implemented in the
MaskGIT generation path; it therefore cannot affect this product-path screen.
The next experiment keeps the all-family margin recipe in both arms and changes
only `grammar_equivalence_cache`, directly targeting the observed 27.3-second
compiler cost and 149 shared-domain misses.

This is fixture evidence only: `n=3<20`, AST and canonical equality are zero,
and held-out, adversarial, OOD, and full Rico were not run. Both arms use seed
101826, 22 CPU scratch steps, batch size 2, and exactly 1,608,962 trainable
parameters. Candidate SHA `49aa39ee...394bea`; control SHA
`1e26bec3...2d89f3`. The checkpoints are local no-sync artifacts and are never
reusable, promotable, syncable, or shippable. Lean is
`not_applicable:screening`; no theorem or promotion claim is made.

Campaign v124 adds the cache successor and orders the terminal matrix around
quality, latency, output length, forwards, compiler time, and cache activity so
the next diagnosis remains visible before cell truncation.

Machine evidence:
[`autotrain-cycle-1826-bounded-margin-null.json`](autotrain-cycle-1826-bounded-margin-null.json).
