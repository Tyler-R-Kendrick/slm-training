# Continuous autotrain: 2026-08-04 (session gd6j83) cycle 3 — frozen-arm retry still times out after the metering fix (finding, not a fix)

**Verdict:** infrastructure failure, not scoreable. The `retry_measurement`
replay of the identical frozen cycle-2 arm (`control` and `component-plan`,
seed 100002, `compiler_decode_mode="tree"` on both arms) was re-run after the
decode-cost metering fix in
[`compiler-tree-forced-closure-decode-metering-gap.md`](compiler-tree-forced-closure-decode-metering-gap.md)
(commits `0e27e47`/`3aad7fe`, merged via `6280d2d`). **Both arms still time
out on all 3/3 smoke records** (`decode_timeout_count=3`,
`evaluation_wall_seconds≈58.0`, `effective_decode_timeout_seconds=24.0`), with
`compiler_ms_mean` still ~23,180ms (`component-plan`) / ~23,287ms (`control`)
— essentially unchanged from cycle 2.

## This confirms the metering fix was necessary but not sufficient

Both cycle-2 and cycle-3 arms decode via `compiler_decode_mode="tree"` — the
compiler-tree path, which was **already** wrapped in `timed_ms(..., "compiler_ms")`
before the fix (the metering gap fixed in cycle 2's follow-up was specifically
in the legacy `compiler_decode_mode="off"` path, `exact_forced_token_id`, and
the forced-closure chain). So this retry's `compiler_ms_mean` was already an
honest measurement in both cycle 2 and cycle 3, and it is genuinely ~23s per
record on both arms, both cycles, at seed 100002 — not a
measurement/attribution artifact. This is a **real** decode-cost problem for
this specific seed's fixture records, distinct from (and downstream of) the
metering gap. The earlier symmetric ~5-6x-vs-cycle-1 comparison remains
invalid as a cross-cycle measurement (per the metering-gap doc, cycle 1's
control decoded via a different, previously-unmetered mechanism), but the
absolute ~23s/record cost for seed 100002 itself is confirmed real and
unresolved.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `harness_failure` on both arms;
`primary_metric_unavailable`). Per `sdlc` autotrain-iteration-delivery, no
stack layer opens for this cycle. This is the second consecutive cycle
blocked on this same frozen manifest lineage (cycle 2 original, cycle 3
retry) — not yet the three-consecutive-failures threshold for a hard-block
report, and there is new information each time (cycle 2: symmetric timeout,
cause unknown; cycle 3: confirmed real per-record cost, not a metering
artifact), so the loop continues to the next typed `repair_harness` action
rather than stopping.

## Next steps (routed, not attempted here)

The driver's typed handoff again names a `repair_harness` action against
harness family `model_build`, this time for frozen manifest
`6953396f2355c70665437a02e075d3f8a67d146b7875e36a0b8b1a6ed2f36597`. This
needs actual profiling of *why* the compiler-tree search costs ~23s/record for
these specific seed-100002 fixture records specifically (node/edge counts,
backtracking depth, grammar-ambiguity shape of the records themselves) --
distinct from the metering question already answered. Candidate outcomes: a
genuine algorithmic fix bounding the search for this record shape, or
evidence-based confirmation that this is legitimate compile cost requiring
either a larger `decode_timeout_seconds` for this record complexity class or
exclusion of pathological fixture records from the smoke suite. Do not
attempt a third automatic retry without that investigation.

Machine evidence:
[`continuous-openui-local-gd6j83-c3-retry-still-times-out.json`](continuous-openui-local-gd6j83-c3-retry-still-times-out.json).
