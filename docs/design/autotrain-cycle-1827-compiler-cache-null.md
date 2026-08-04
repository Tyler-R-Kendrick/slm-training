# Autotrain c1827: compiler cache null

**Verdict:** reject `grammar_equivalence_cache` as a material compiler-tree
optimization. Quality and decode work are identical, and the cached arm's 0.67%
p50 improvement is below the preregistered 5% efficiency floor.

| Arm | Params | Structure | MPR | Recall | Binder F1 | Fidelity | Reward | Tokens | Forwards | Compiler ms | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| un-cached margin control | 1,608,962 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 60 | 11 | 6659.90 | 2870.15 |
| cached margin | 1,608,962 | .46417 | .3333 | .25 | .6333 | .5278 | .8073 | 60 | 11 | 6622.47 | 2850.97 |

Both arms use 8,192 compiler-prefill tokens, 2,816 canvas tokens, and build
91,845 completion states. The candidate effective config records
`grammar_equivalence_cache=true`, but both arms report zero shared-domain hits
and 45 misses. Compiler-tree already shares completed domains within the batch;
the tested equivalence cache does not reuse these request-specific prefixes.

The next experiment retains the all-family margin recipe and changes only the
certified compiler draft window from 8 to 16. Longer grammar-verified spans may
amortize completion-forest construction and ranking across fewer decisions;
legality, deterministic authority, model parameters, and learned scores remain
unchanged.

This remains fixture evidence: `n=3<20`, MPR and recall miss their gates, AST
and canonical equality are zero, and held-out, adversarial, OOD, and full Rico
were not run. Both arms use seed 101827, 20 CPU scratch steps, batch size 2, and
1,608,962 trainable parameters. Candidate SHA `45d1ee73...c739292`; control SHA
`9eec7fe6...cdbb0d4`. Checkpoints are local no-sync artifacts and are never
reusable, promotable, syncable, or shippable. Lean is
`not_applicable:screening`.

Machine evidence:
[`autotrain-cycle-1827-compiler-cache-null.json`](autotrain-cycle-1827-compiler-cache-null.json).
