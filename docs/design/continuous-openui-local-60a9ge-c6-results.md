# Autotrain c6 (continuous-openui-local, session 60a9ge): repeated hard blocker, escalating

**Verdict:** infrastructure failure, not scoreable, and now a **confirmed
repeated hard blocker** (3 consecutive cycles: 4, 5, 6). Both `control` and
`component-edge` hit `smoke:decode_timeout_count=3` again — compiler time
23218.7ms / 23056.4ms mean, vs the policy-configured
`screening_decode_timeout_seconds=8`.

## Why this isn't fixable by another retry

`thrash_timing.json` for this cycle shows `clamp_bound=0.0`: the
policy-configured `8s` is the binding constraint, not a wall-fit squeeze —
the wall-fit ceiling would actually allow up to `14s`
(`arm_wall_seconds=70`, minus `min_train_floor_seconds=20` and
`eval_overhead_seconds=8`, divided by `smoke_n=3`). But even `14s` would not
be enough: the observed per-document compiler cost has been a consistent
`~23–24s` across cycles 4, 5, and 6. Three documents at that rate
(`~69–72s`) alone would consume the entire `70s` arm wall, leaving no room
for the `20s` train floor or `8s` overhead — a structural capacity mismatch
for this compiler-tree-heavy recipe (`--grammar-completion-bounds`,
component-edge decode heads) on this host's CPU throughput, not something a
larger `decode_timeout_seconds` alone can fix without also raising the arm
wall, which would collide with the repository's `MAX_RUN_MINUTES=3` hard
cap.

## Why this cycle does not claim a repair

Cycles 4 and 5 acknowledged `repair_harness` as `completed` with
"no code change needed" evidence, which is what let the driver's
`max_consecutive_frozen_replays` counter reset and permit another identical
retry each time. Doing that a third time in a row, with the exact same
signal recurring, would not be an honest repair — it would just be relabeling
a genuine unresolved capacity blocker as fixed to keep the loop spinning.
Per `autotrain`'s repeated-blocker rule (same hard blocker failing three
consecutive cycles with no new decision made), this cycle's `repair_harness`
action is acknowledged with **`status=blocked`**, not `completed` — the
evidence commit is this documentation, not a fix.

This is a genuine `improve-openui-harnesses` recalibration question (per the
policy file's own comment: *"Incomplete rate outside [low,high] means
recalculate recipe/budget, not ad-hoc wall++"*), not a docs-only self-heal:
either the screening recipe needs a cheaper CPU-only decode path, or this
compiler-tree-heavy comparison needs to be scoped to faster (GPU) hosts for
screening, or `screening_thrash_steps`/`min_train_floor_seconds` need
rebalancing against observed p95 compile cost. None of those are safe to
decide unilaterally inside a docs-only cycle, so it is left open rather than
faked closed.

## SDLC Phase A

**Non-positive** (`harness_failure` / `measurement_incomplete` on both arms,
3rd consecutive repeat). Per `sdlc` autotrain-iteration-delivery, no stacked
PR layer is opened — local commit and docs only.

Next: escalate to `improve-openui-harnesses` for a real thrash-timing
recalibration before any further automatic retry of this exact arm/recipe
pairing on CPU-only hosts.

Machine evidence:
[`continuous-openui-local-60a9ge-c6-results.json`](continuous-openui-local-60a9ge-c6-results.json).
