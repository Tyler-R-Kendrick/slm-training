# Autotrain c1718: completion-branch direct-feed signal

**Verdict:** fixture/scratch measurement incomplete. Verified stage reuse skipped
training for both arms and wrote explicit manifest/checkpoint receipts. The control
completed honest smoke evaluation and failed its ship gates. The batch-size-1
candidate still timed out in constrained decoding before any scoreable evaluation.
No quality or latency comparison is authorized.

## Result matrix

| Arm | Reused train source | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | c1717 control, lineage depth 1, checkpoint `e38a18ad…7c54ee` | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 4,357.37 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | c1716 batch1, lineage depth 2, checkpoint `3ef5056a…14dc64` | No metrics | Evaluation timed out in terminal-witness completion-forest branch advance | Frozen replay required |

Both immutable reuse receipts are embedded in the
[machine-readable record](autotrain-cycle-1718-direct-feed-timeout.json) under
`arms[].training.reuse_receipt`. The control receipt binds train summary
`aa12e745…707132` to ordered manifest `dbacf796…f3c42c`; the batch1 receipt binds
train summary `0d78ae37…494668` to ordered manifests
`779b44e3…999113` then `422ce7ff…5657b`. Each arm also records its checkpoint SHA,
lineage depth, and `executed=false`. Evaluation wrote only to c1718 run namespaces.
The handoff remains `inconclusive`, `ship_state=blocked`, and binds its retry to
manifest `1ac8dc0acefc52f86ba7e5604045c2ada0b4a4e6623030921ccb74e6fb3e8779`.

## Diagnostic signal and repair

With retraining and unconditional AST parsing removed, the candidate interrupt trace
ends in:

```text
terminal_witness
  -> outgoing
  -> _build_openui_completion_forest_direct
  -> branch.advance_checked
```

The completion session already has a checked static DSL-token-to-Lark-terminal map
and uses it for state transitions. Completion-forest construction nevertheless sent
every candidate and deterministic forced token through text lexing. The repair uses
`feed_token_id` first and retains `advance_checked` as the canonical fallback when
the junction is ambiguous or the token is unsupported. Grammar rejection restores
the branch state exactly; EOS/finalize authority and terminal-witness budgets are
unchanged.

A regression proves a representative reachable component forest uses zero text
advances, while the full 224-test compiler-decode module preserves behavior. This is
a code-path/count result, not an end-to-end speed or TPS claim. The next exact frozen
replay is authoritative.

## Provenance and next-run priorities

Eval commit: `09ab77cb7221188b0d16838f4d25f9cd732ed42c`
(`model.twotower=v271`, `harness.autoresearch.experiment_campaign=v43`). No new
checkpoint was created or promoted. The direct-feed repair is `model.twotower=v272`.
Machine-readable evidence is in
[`autotrain-cycle-1718-direct-feed-timeout.json`](autotrain-cycle-1718-direct-feed-timeout.json).

1. Replay the exact frozen evaluations using the same verified checkpoints.
2. Require both fresh authoritative scoreboards before comparing the arms.
3. If terminal-witness still exceeds the cap, use the next interrupt trace and
   exact session counters to target duplicate branch/state work; never widen grammar
   authority, proof budgets, or deadlines.
4. Keep Lean formal preflight on promotion cadence. c1718 supplies no optimum or
   promotion evidence.
