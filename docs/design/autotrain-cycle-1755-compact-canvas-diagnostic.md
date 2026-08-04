# Autotrain c1755: compact-canvas diagnostic

**Verdict:** reject. Compact active canvas is an exact smoke quality and loss
null, while its completed p50 latency is 6.08% slower than the matched control.

| Arm | Params | n / complete / timeout | Parse | Binder F1 | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 3 / 3 / 0 | 1 | .6333 | .3333 | .17417 | .25 | 1,120.45 ms | 15.4050 / 2.606 s |
| compact canvas | 1,608,962 | 3 / 3 / 0 | 1 | .6333 | .3333 | .17417 | .25 | 1,188.59 ms | 15.4050 / 2.433 s |

Both size-matched 22-step CPU scratch arms used seed 101755, batch size 2,
strict compiler-tree constrained decoding, `honest_slot_contract=True` during
evaluation, and completed all three smoke documents. Compact canvas changes
neither quality nor loss, saves 6.63% training wall, but adds 68.14 ms to
completed decode p50. This is a fixture diagnostic, not production evidence.

AgentV evidence is complete and both `--ship-gates` verdicts fail: smoke has
only `n=3`, misses meaning, structure, recall, AST BEq, and canonical BEq bars,
and the held-out, adversarial, OOD, and `rico_held` suites were not run. RL
remains locked. Lean is `not_applicable:screening`; no promotion theorem was
claimed and neither local checkpoint may be reused, promoted, synced, or
shipped.

Next: recompute the successor under current `origin/main` policy. Prioritize a
distinct quality hypothesis; do not combine two rejected runtime-null levers
merely because the pre-integration c1755 handoff suggested `both`.

Machine evidence:
[`autotrain-cycle-1755-compact-canvas-diagnostic.json`](autotrain-cycle-1755-compact-canvas-diagnostic.json).
