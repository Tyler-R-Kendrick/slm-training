# Continuous autotrain: 2026-08-04 cycle 1 (new loop_id `continuous-openui-local-2`) — null delta, fresh authority confirmed healthy

**Loop:** `continuous-openui-local-2` (new lineage — see below for why)
**Campaign:** `continuous-loop-20260804-continuous-openui-local--c94ddb78-c1`
**Integration commit:** `f1375789` (post-HX4/HX5, post decode-metering-gap fix, post branch-point-cost finding)

## Why a new loop_id

The prior loop (`continuous-openui-local`) is permanently blocked at cycle 4:
[`continuous-openui-local-gd6j83-c4-frozen-arm-authority-stale.md`](continuous-openui-local-gd6j83-c4-frozen-arm-authority-stale.md)
found that its frozen checkpoint's `completion_artifact` identity no longer
matches the installed grammar/tokenizer authority after merging
`origin/main`'s unrelated HX4/HX5 commit (`807d4f8`) — `require_checkpoint_completion_artifact`
correctly, by design, refuses to load it (the same class of protection as
I3/I6: never silently reuse a stale decode authority). The supervised
driver's own predecessor-prerequisite check will not advance past an
unacknowledged-as-completed `repair_harness` action for that campaign under
the same `loop_id`, and rightly so — there is nothing to repair, so
acknowledging it as `completed` would be dishonest. Rather than fight that
correct safety gate, this session starts a fresh `loop_id` to keep training
under the current authority.

## Result

| Arm | structural_similarity | parse_rate | binder_reference_f1 | compiler_ms_mean | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | .05750 | 1.0 | .63333 | 3607.97 | 3822.31 |
| bounds | .05750 | 1.0 | .63333 | 3467.21 | 3672.75 |

Primary delta `0.0` — the same null result on the `bounds` (knob-rotation)
arm already seen in two prior sessions (`ts5ofk`, `gd6j83`), now confirmed a
third time, this time under the post-HX4/HX5 authority. Both arms decode
cleanly within budget (`compiler_ms_mean` ~3.5-3.6s/record, no decode
timeouts) — consistent with the gd6j83-c4 diagnosis that the earlier load
failure was specifically an authority mismatch on a *stale* checkpoint, not
a regression affecting fresh training/eval under current code.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). No stack layer opens.

## Next priorities

1. Screen the `component-plan` hypothesis next under this fresh authority —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle.
3. Rotate thrash recommendation across the lever bank rather than repeating
   `bounds`-only.

Machine evidence:
[`continuous-openui-local-2-c1-results.json`](continuous-openui-local-2-c1-results.json).
