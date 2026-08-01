# Autotrain c1719: CPU compiler-prefill signal

**Verdict:** fixture/scratch measurement incomplete. Both training stages were
reused from immutable, size-matched 1,608,962-parameter checkpoints. The control
completed honest smoke evaluation and failed ship gates. The batch-size-1 arm still
timed out before a scoreable evaluation, so no arm comparison or promotion is
authorized.

## Result matrix

| Arm | Reused train source | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | c1717 control, lineage depth 2, checkpoint `e38a18ad…7c54ee` | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 4,655.07 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | c1716 batch1, lineage depth 3, checkpoint `3ef5056a…14dc64` | No metrics | Evaluation timed out in a compiler-tree denoiser prefill | Frozen replay required |

The exact source train-summary, checkpoint, and ordered manifest digests for both
reuse receipts are embedded under `arms[].training.reuse_receipt` in the
[machine-readable record](autotrain-cycle-1719-cpu-prefill-timeout.json). Evaluation
wrote only to c1719 run namespaces. The handoff remains `inconclusive`,
`ship_state=blocked`, and binds its retry to manifest
`890c04856c3fd61c35ffc1074841440136db8e88a7d0a961ac67a685eb815ffe`.

## Diagnostic signal

The c1718 direct-feed change moved the interrupt out of grammar search. c1719 ends
in:

```text
_compiler_ltr_decode_batch
  -> _denoiser_hidden
  -> DenoiserTower.encode
  -> TransformerBlock.mlp
  -> Linear
```

This does not prove one forward is individually slow. It proves only that the
remaining wall is consumed while the exact tree scorer evaluates neural prefill
states. The completed control recorded 58 prefill states in 58 batches, so its
forests exposed one ambiguous state at a time. The stopped candidate emitted no
partial counter row; its exact state count remains unknown.

## Exact-checkpoint CPU prefill probe

The bounded local probe used the candidate checkpoint, 12 Torch CPU threads, a
256-token canvas, nine context tokens, one warmup, and five timed backbone forwards
per row. Lower milliseconds per state means better packing throughput.

| Prefill states | Median batch ms | Median ms/state | Signal |
| ---: | ---: | ---: | --- |
| 1 | 197.764 | 197.764 | Under-filled |
| 2 | 137.050 | 68.525 | Better packing |
| 4 (current auto) | 145.031 | 36.258 | Baseline |
| 8 | 160.052 | 20.006 | 1.81× baseline state throughput |
| 16 | 221.460 | 13.841 | 2.62× baseline state throughput |

Batching 16 states takes 1.53× the wall time of batching four while evaluating four
times the exact states. This is a fixture/scratch backbone microbenchmark, not an
end-to-end decode, suite-latency, or TPS result.

## Repair and next-run priorities

The repair raises only the automatic CPU tree-prefill pack from four to sixteen
states. Explicit `compiler_prefill_max_states` and token budgets remain authoritative;
GPU packing remains 32. A regression checks the default `16 + 15` packing of a
31-parent trie and proves the selected path matches four-state packing on the same
model. Grammar membership, candidate scores, model parameters, deadlines, and
fail-closed validation are unchanged.

1. Replay the identical frozen c1719 manifests; only fresh scoreboards authorize a
   comparison.
2. Require candidate completion plus counters for prefill batches, states, backbone
   time, and total latency before claiming an end-to-end gain.
3. If it still times out, persist interrupt-safe partial decode counters before
   considering a semantic search-policy experiment.
4. Keep Lean and Lean-formal CI required. c1719 produced no theorem optimum,
   checkpoint, or promotion evidence, so no model-card change is required.

Eval commit: `36775a848d95cebee1686589dc4d3d8a473b7f82`
(`model.twotower=v272`). The scheduling repair is `model.twotower=v273`.
