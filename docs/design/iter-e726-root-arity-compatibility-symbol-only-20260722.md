# E726 root-reference arity compatibility failure

**Date:** 2026-07-22  
**Decision:** invalidate the experiment checkpoint; fail closed before future runs  
**Evidence:** [`iter-e726-root-arity-compatibility-symbol-only-20260722.json`](iter-e726-root-arity-compatibility-symbol-only-20260722.json)

## Question

E723's slot-owner signal can choose component identities but still closes the
root after too few reachable bindings. Earlier choice-codec evidence showed
that learned root-reference arity changes count decisions. E726 attempted to
combine those signals on the current lexer-based symbol-only v2 path.

## Completed but invalid arm

The local CPU scratch command completed 140 steps in 81.20 seconds under
`max_wall_minutes=2` on the exact 141-record strict snapshot (manifest
`78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`).
It requested E723's plan, edge, and slot-owner recipe plus root-reference arity
loss/decode weight 1. The local no-sync checkpoint SHA is
`d84148fe327c18dee6a4ad4957b1b23499e17ae364c79a97cdc8150503a1b91b`.

This is not valid experiment evidence. Root-reference arity and identity heads
were implemented only for the choice tokenizer. With `output_tokenizer=lexer`,
the requested nonzero weights were silently stored in configuration while no
head, loss, or decode intervention existed. E726's final loss 9.1764, primary
loss 5.0753, slot-owner metrics, token exposure, and all 137 checkpoint tensors
are exactly identical to E723. The differing checkpoint file hash comes from
stored configuration metadata, not learned weights. No evaluation was run
because the treatment did not exist.

## Harness correction

The canonical lever registry now declares the supported output tokenizer for
all four root-reference activation weights. Shared model-build and TwoTower
configuration validation consumes that registry and rejects an enabled
choice-only lever for lexer or compositional output before model construction
or run-directory creation. The actual training CLI now exits with a typed
`require output_tokenizer='choice'` error and writes no artifacts. Focused
applicability also declares the owning model family, so a non-TwoTower model
cannot accept these weights. Registry and configuration coverage passes 37/37.

Component versions advance to `config.levers` v2 and `model.twotower` v189.
This fixes the harness/config contract; it does not hide or reinterpret the
failed E726 run.

## Disposition

Keep the checkpoint only as invalidated provenance. Do not evaluate, sync,
promote, serve, or compare it as a model result. Future symbol-only structural
work must use a lever implemented for the lexer path or explicitly add that
capability with tests before launching training.
