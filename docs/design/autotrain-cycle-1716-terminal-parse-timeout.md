# Autotrain c1716: terminal AST-parse timeout

**Verdict:** fixture/scratch measurement incomplete. The exact current-main replay
reproduced both c1715 checkpoint hashes. The matched control completed honest smoke
evaluation and failed its ship gates; the batch-size-1 candidate again completed
training but timed out in constrained decoding before any scoreable evaluation.
Neither checkpoint is reusable, promotable, synced, or ship evidence.

## Result matrix

| Arm | Recipe | Params | Smoke result | Gate / completion | Disposition |
| --- | --- | ---: | --- | --- | --- |
| control | CPU, 80 steps, batch 2, seed 101710 | 1,608,962 | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 3,705.20 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | CPU, 80 steps, batch 1, seed 101710 | 1,608,962 | No metrics | Evaluation timed out in `terminal_witness`; incomplete | Frozen replay required |

The control scoreboard is authoritative despite its honest gate rejection. The
candidate has no scoreboard, so this cycle cannot attribute a batch-size quality or
latency effect. The handoff remains `inconclusive`, `ship_state=blocked`, and carries
an exact `retry_measurement` action bound to frozen manifest
`422ce7ff5031fd20f801d99709d367274a694741931795f99527efc64335657b`.

## Diagnostic signal and repair

The c1715 tokenizer-kind cache moved the observed interrupt deeper:

```text
terminal_witness
  -> outgoing
  -> _build_openui_completion_forest_direct
  -> _generated_ast_is_complete
  -> lang_core.parse
```

Completion-forest construction called official AST completeness parsing at every
grammar state, although the result only controls EOS and terminal continuation when
`$END` is legal. The repair invokes that authoritative parser only at `$END` states.
Nonterminal states cannot admit EOS, so skipping their parse is exact: it does not
widen candidates, substitute a parser, change terminal-witness budgets, or weaken
finalize validation. A regression test proves zero AST parse calls for a nonterminal
prefix and retains official parsing plus EOS admission for a complete terminal prefix.

This is a code-path/count reduction, not an end-to-end speed or TPS claim. The next
authoritative test is another identical frozen control/batch1 replay after merge.

## Checkpoints and provenance

| Arm | Local checkpoint SHA-256 | Status |
| --- | --- | --- |
| control | `e38a18adafcf2812e30f9f4ef748962b43834dc08fe57a99b43851b5fa7c54ee` | no-sync fixture; reject |
| batch1 | `3ef5056af61ed73d19b1ca90ef0ddb98f05389b2145a83e75f00360cad14dc64` | no-sync fixture; incomplete; reject |

Train/eval commit: `3c52bd297582880a87e7ef3b280c5cd3b43ff1ce`
(`model.twotower=v270`, `harness.model_build.eval=v71`). The terminal-only AST
parse repair is `model.twotower=v271`. Machine-readable evidence is in
[`autotrain-cycle-1716-terminal-parse-timeout.json`](autotrain-cycle-1716-terminal-parse-timeout.json).

## Next-run priorities

1. Replay the exact frozen candidate and matched control after this repair; preserve
   seed, steps, batch sizes, endpoints, gates, and stopping rules.
2. Require both authoritative scoreboards before comparing the arms.
3. If exact terminal-witness search still exceeds the wall cap, measure the next
   observed call site and optimize only request-local exact work; do not widen the
   grammar, proof budget, or deadline.
4. Keep Lean formal preflight on promotion cadence. c1716 is screening evidence and
   supplies no optimum or promotion claim.
