# Autotrain c1715: terminal-witness timeout

**Verdict:** fixture/scratch measurement incomplete. The matched control completed
honest smoke evaluation and failed its ship gates; the batch-size-1 candidate
completed training but timed out in constrained decoding before any scoreable
evaluation. Neither checkpoint is reusable, promotable, synced, or ship evidence.

## Result matrix

| Arm | Recipe | Params | Smoke result | Gate / completion | Disposition |
| --- | --- | ---: | --- | --- | --- |
| control | CPU, 80 steps, batch 2, seed 101710 | 1,608,962 | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 3,884.32 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | CPU, 80 steps, batch 1, seed 101710 | 1,608,962 | No metrics | Evaluation timed out in `terminal_witness`; incomplete | Frozen replay required |

The outer experiment exit was not a model comparison. The control's evaluator exit
8 is a typed, complete gate rejection backed by `scoreboard.json`; the candidate is
the only incomplete arm. The campaign handoff therefore remains `inconclusive` with
`ship_state=blocked` and an exact `retry_measurement` action.

## Diagnostic signal and repair

The candidate interrupt trace ended in:

```text
terminal_witness
  -> outgoing
  -> _build_openui_completion_forest_direct
  -> DSLNativeTokenizer.kind_ids
```

`kind_ids` rebuilt the same kind membership by scanning the frozen vocabulary on
every call. The repair caches immutable kind membership on the tokenizer, returns a
fresh set to preserve caller behavior, restores the cache lazily for older serialized
checkpoints, and removes redundant set copies at the observed compiler call site.
Grammar legality, terminal-witness authority, node budgets, deadlines, and Lean
promotion preflights are unchanged.

Fixture microbenchmark (`python -m timeit -n 10000 -r 7`, local CPU):

| Operation | Best time/call | Relative |
| --- | ---: | ---: |
| Historical direct vocabulary scan | 9.28 us | 1.00x |
| Cached `kind_ids(BIND)` with fresh-set return | 0.512 us | 18.13x |

This 94.5% subroutine reduction is a hotspot microbenchmark, not an end-to-end decode
or tokens-per-second claim. The next authoritative test is the identical frozen
control/batch1 replay on merged current main.

## Checkpoints and provenance

| Arm | Local checkpoint SHA-256 | Status |
| --- | --- | --- |
| control | `e38a18adafcf2812e30f9f4ef748962b43834dc08fe57a99b43851b5fa7c54ee` | no-sync fixture; reject |
| batch1 | `3ef5056af61ed73d19b1ca90ef0ddb98f05389b2145a83e75f00360cad14dc64` | no-sync fixture; incomplete; reject |

Historical train/eval commit: `b5134b90532986384fbd34c9fd609b2681bfe390`
(`model.twotower=v269`, `harness.model_build.eval=v71`). The cache repair is
`model.twotower=v270`; the status projection is
`harness.autoresearch.experiment_campaign=v42`. Machine-readable evidence is in
[`autotrain-cycle-1715-terminal-witness-timeout.json`](autotrain-cycle-1715-terminal-witness-timeout.json).

## Next-run priorities

1. Replay the exact frozen candidate and matched control after this repair; do not
   change seed, steps, batch sizes, endpoints, gates, or stopping rules.
2. Require both scoreboards before attributing any batch-size quality or latency
   effect.
3. If terminal-witness time still dominates, profile completion-forest construction
   under the same manifest and investigate only exact request-local caches; do not
   widen the grammar, node budget, or deadline.
4. Keep Lean formal preflight on promotion cadence. This screening timeout supplies
   no formal optimum or promotion evidence.
