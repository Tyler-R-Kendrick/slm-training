# Autotrain c1720: uncached token-map fingerprint signal

**Verdict:** fixture/scratch measurement incomplete. Both 1,608,962-parameter
training stages were reused from immutable, size-matched checkpoints. The control
completed honest smoke evaluation and failed ship gates. The batch-size-1 arm
timed out before a scoreable evaluation, so no arm comparison or promotion is
authorized.

## Result matrix

| Arm | Reused train source | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | c1717 control, lineage depth 3, checkpoint `e38a18ad…7c54ee` | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 3,838.22 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | c1716 batch1, lineage depth 4, checkpoint `3ef5056a…14dc64` | No metrics | Evaluation wall timeout while constructing an unused tokenizer cache fingerprint | Frozen replay required |

The exact source train-summary, checkpoint, and ordered manifest digests for both
reuse receipts are embedded under `arms[].training.reuse_receipt` in the
[machine-readable record](autotrain-cycle-1720-token-map-timeout.json). Evaluation
wrote only to c1720 run namespaces. The handoff remains `inconclusive`,
`ship_state=blocked`, and binds its retry to manifest
`efa71f58de59176fb5a7a84b9a791369eff9a16141c6648d91a0b21558e6eca2`.

The control's p50 is lower than c1719's 4,655.07 ms, but each value is one
three-document fixture replay. This is neither a powered comparison nor evidence
that the c1719 CPU packing change caused the difference.

## Diagnostic signal

The supervised timeout interrupt ended in:

```text
terminal_witness
  -> outgoing
  -> _build_openui_completion_forest_direct
  -> _decision_kind
  -> _grammar_terminal_kind
  -> allowed_id_set
  -> sorted(tokenizer.token_to_id.items())
```

`_grammar_terminal_kind` requests the default uncached exact terminal mask.
`allowed_id_set` nevertheless sorted the full token map and hashed it before
checking `use_cache=False`. That fingerprint had no consumer and could recur for
every decision terminal. The trace identifies the interrupted operation; it does
not quantify its share of total decode wall time.

## Repair and next-run priorities

The repair constructs the content-sensitive tokenizer fingerprint only when
`use_cache=True`. Cached callers retain the existing mutation-safe key. Uncached
callers still compute the exact mask afresh, and grammar membership, model scores,
parameters, deadlines, and fail-closed validation are unchanged. A regression uses
a mapping whose `items()` raises to prove the uncached path never touches cache-key
machinery. The focused grammar suite also corrected a stale assertion to enforce
I2's now-exact singleton `root -> =` completion.

1. Replay the identical frozen c1720 batch1 manifest; only a complete fresh
   scoreboard can authorize an arm comparison.
2. If the arm still times out, use the next supervised interrupt plus
   interrupt-safe partial decode counters to select the next implementation repair.
3. Measure repeated exact-mask/fingerprint call counts before enabling broader
   caching; do not substitute a heuristic or stale tokenizer authority.
4. Keep Lean and Lean-formal CI required. c1720 produced no theorem optimum,
   checkpoint, or promotion evidence, so no model-card change is required.

Eval commit: `58e5fa2856455ac0a50a981a0c9422612d1456b9`
(`model.twotower=v273`). The exact no-cache repair is `model.twotower=v274`.
