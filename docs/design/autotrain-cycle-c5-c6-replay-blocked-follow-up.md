# Autotrain c5/c6: two distinct blockers, follow-up needed (finding, not a fix)

**Status:** blocked, needs dedicated follow-up session. Do not keep
re-attempting automatically.

## Blocker 1 — dual-arm decode timeout (c5)

Documented in
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.md):
the c4 champion's fresh-seed confirmation (seed `100005`, 1,755,760 params)
finalized 3/3 smoke records inside a typed decode timeout on **both** the
control and confirm arms, after the same size decoded successfully in
~6.5-7.2s per arm one cycle earlier at a different seed. Root cause (seed-
dependent decode pathology vs. sandbox CPU/wall-budget headroom) was not
determined. A drafted routing fix (auto-retire symmetric dual-arm timeouts)
was correctly reverted: it violated the existing, deliberate
`tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`
contract, which requires this exact case to keep demanding real repair
rather than silently retiring.

## Blocker 2 — frozen-replay cannot resurrect a confirmation-cycle arm (c6)

Attempting cycle c5's own recommended next step — "replay the identical
frozen confirm arm" — surfaced a second, distinct, lower-level bug:
`_apply_frozen_replay` (`scripts/run_autotrain_continuous.py:5225`) maps an
old candidate arm's ID suffix to a registered slug in `_SCREENING_ARM_BANK`
to build the successor arm's new experiment IDs. Confirmation-cycle arms are
suffixed `-confirm`, which is **not** in `_SCREENING_ARM_BANK` (a catalog of
regular screening-hypothesis slugs like `bounds`, `component-plan`, etc.), so
the automatic frozen replay raises:

```
RuntimeError('unsupported automatic frozen replay arm: c20260803-continuous-openui-local-8c0b60dd-c5-confirm')
```

This means the driver's own recommended remedy for blocker 1
(retry_measurement on the frozen `-confirm` arm) is currently **inexecutable**
through the automatic continuous-cycle path. This is a real, narrow gap: the
frozen-replay arm-slug lookup needs to also recognize confirmation-cycle
suffixes (or `_apply_frozen_replay` needs an explicit path for confirmation
arms distinct from screening arms).

## Recommendation

Both blockers need `improve-openui-harnesses` attention in a dedicated
session with room to profile compiler-tree decode at seed `100005` (blocker
1) and to extend `_apply_frozen_replay`'s slug registry or add a
confirmation-arm-specific branch (blocker 2), each with its own regression
test. Do not attempt further speculative fixes to either without that
investigation; in particular, do not resurrect the auto-retire-on-symmetric-
timeout routing change for blocker 1 — it was tried and correctly reverted.

The c4 efficiency finding
([`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md))
remains queued (`champ-continuous-openui-local-4-2694d77fc99953e4`),
unconfirmed, pending resolution of blocker 2 at minimum.
