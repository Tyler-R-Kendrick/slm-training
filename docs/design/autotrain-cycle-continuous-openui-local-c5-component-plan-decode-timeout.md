# Autotrain continuous-openui-local c5: component-plan decode timeout (soft failure, not a harness defect)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

**Verdict:** cycle 5 of loop `continuous-openui-local` (campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c5`) tested the
driver's ranked `component-plan` quality hypothesis
(`component_plan_decode_weight=1.0`, `compiler_decode_mode=tree`,
grammar-constrained decode, `generate_max_attempts=3`) against a fresh
`control` (`compiler_decode_mode=off`). The control completed and honestly
rejected ship gates on fixture-scale `n=3` (same pattern as cycle 4). The
`component-plan` candidate hit a **decode timeout on all 3 documents**
(`completed_document_n=0`, `incomplete_document_n=3`,
`decode_timeout_count=3`) — `measurement_incomplete`, not a model result.

## Root cause

Not a harness code defect. `compiler_decode_mode=tree` with
`component_plan_decode_weight=1.0` (grammar-constrained tree decode,
`generate_max_attempts=3`) is substantially more expensive per token than
the `compiler_decode_mode=off` baseline every prior cycle in this loop
instance used. On this session's CPU-only sandbox host,
`MAX_RUN_MINUTES=3` caps the whole experiment, leaving only ~29s of
evaluation wall budget for `n=3` documents against a 24s per-document decode
timeout — not enough headroom for tree-constrained decode to finish even one
document. No exception, no stack trace, no incorrect output: the decode path
is simply slower than the available wall budget at this hardware/knob
combination.

## Classification

Soft failure per `continuous.md`: *"ship gates fail on fixture n, null lever
deltas, timeouts never stop the loop."* No code change made. Per the
self-heal guidance (*"change knobs and re-run"* when evidence does not name
a canonical harness family), the correct response is to steer the next
screening cycle away from `compiler_decode_mode=tree` candidates at this
wall cap rather than retry the identical timing-bound arm or force a
harness-code repair for a decode path that is working as designed, just
slow.

## Next priority

Prefer `compiler_decode_mode=off` (or otherwise wall-budget-compatible)
candidates for CPU-sandboxed screening cycles in this loop instance; reserve
`compiler_decode_mode=tree` component-plan comparisons for hosts/wall caps
that can complete at least one full grammar-constrained decode.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c5-component-plan-decode-timeout.json`](autotrain-cycle-continuous-openui-local-c5-component-plan-decode-timeout.json).
