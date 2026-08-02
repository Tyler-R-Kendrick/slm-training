# Autotrain c1721: interrupt-safe decode telemetry

**Verdict:** fixture/scratch measurement incomplete. Both 1,608,962-parameter
training stages were reused from immutable, size-matched checkpoints. The control
completed honest smoke evaluation and failed ship gates. The batch-size-1 arm again
hit the stage wall before a scoreable evaluation, so no arm comparison or promotion
is authorized.

## Result matrix

| Arm | Reused train source | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | c1717 control, lineage depth 4, checkpoint `e38a18ad…7c54ee` | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 2,930.60 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | c1716 batch1, lineage depth 5, checkpoint `3ef5056a…14dc64` | No metrics | Supervisor interruption during recursive terminal-witness work | Frozen replay required |

The exact source train-summary, checkpoint, and ordered manifest digests for both
reuse receipts are embedded under `arms[].training.reuse_receipt` in the
[machine-readable record](autotrain-cycle-1721-interrupt-telemetry.json). The
handoff remains `inconclusive`, `ship_state=blocked`, and binds its retry to manifest
`5f87b83ea36f943eb437c5dce5f5f08a12f9d9c741f8a055fba3e5dad00ea85d`.

The control's p50 is lower than c1720's 3,838.22 ms, but each value is one
three-document fixture replay. It is not a powered latency comparison and cannot be
attributed to either implementation repair.

## Diagnostic signal

The c1720 no-cache repair moved the sampled interrupt past token-map sorting. c1721
ends in:

```text
terminal_witness
  -> _eval
  -> outgoing
  -> for path in forest.paths
  -> tuple(int(token_id) for token_id in path.token_ids)
```

The tuple conversion is a trivial leaf operation and a single interrupt sample does
not establish it as the bottleneck. The completed control supplies useful scale:
1,395 terminal-witness states, 1,112 completion edges, and 58 neural prefill states
for three documents. The stopped arm lost all corresponding partial counters because
stats were folded and persisted only after normal return. That observability loss is
the next harness defect.

## Repair and next-run priorities

The repair keeps one canonical `DecodeStats` path. Exceptional exits attach the live
bucket; each forest call folds completion-session deltas in `finally`; the evaluator
atomically writes a version-stamped `DecodeProgressV1` sidecar; and autoresearch
ingests only a fresh sidecar into stopped-stage telemetry. The artifact is explicitly
non-scoreable and is removed after canonical evaluation succeeds. Grammar authority,
model scores, parameters, deadlines, gates, and promotion policy are unchanged.

1. Replay the identical frozen c1721 manifest and require the stopped or completed
   outcome to expose exact witness, edge, prefill, forward, and timing counters.
2. Use those counters to decide whether the next experiment targets completion
   search, grammar-domain reuse, or neural prefill; do not infer a hotspot from one
   sampled stack line.
3. Keep the control size-matched and retain honest missing-suite and quality failures.
4. Keep Lean and Lean-formal CI required. c1721 produced no theorem optimum,
   checkpoint, or promotion evidence, so no model-card change is required.

Eval commit: `6f38011faff5913f564fbe7969b934b1c580320c`
(`harness.autoresearch.experiment_campaign=v43`,
`harness.model_build.eval=v71`, `model.twotower=v274`). The telemetry repair bumps
those components to `v44`, `v72`, and `v275`, respectively.
