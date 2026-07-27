# SLM-434 (LAR0-07): SLM-305/308/310 model-work port decision

Follow-up required by SLM-431 (LAR0-06)'s stop-rule finding. Full machine
evidence: `iter-slm434-lar0-07-model-work-port-20260727.json`.

## Decision

Port `value_label_mode` and `stop_slot_accounting` (route **a**) rather than
re-express the SLM-317 historical/improved arms against main's config without
them (route **b**).

## Why

Route (b) is a genuine no-op for the **historical** arm only: main's
`training_loss`/`_decode_one` already implement exactly `mutation_count`
value labels (`1 - applied/(max_chain+1)`) and `legacy` STOP-slot accounting
(every STOP proposal consumes an `expand_per_state` slot) — confirmed by the
full pre-existing test suite passing **unmodified** after the port. But the
**improved** arm inherently needs the `bounded_distance` oracle and
`corrected` STOP accounting; there is no way to re-express that arm without
the knobs. Per SLM-434's own contract ("if (b) changes what the arms
measure, prefer (a)"), route (a) was mandatory for the improved arm, so this
port covers both knobs rather than mixing routes per-arm.

## Scope

Ported (needed by the landed SLM-317/SLM-431 harness):

- `TreeEditDiffusionConfig.value_label_mode` (`mutation_count` default /
  `bounded_distance`) and `.stop_slot_accounting` (`legacy` default /
  `corrected`), both **default-off** — the default reproduces main's
  pre-existing behavior exactly.
- `training_loss`'s `bounded_distance` branch: value targets from the ported
  SLM-308 oracle, UNKNOWN excluded from the value MSE (never coerced).
- `_decode_one`'s `corrected` branch: STOP consumes decode budget only when
  its frozen candidate is retained on the beam.
- `TreeEditSpace.apply` gains an accept-only optional `reason` out-list (a
  single generic marker) so the already-landed harness's `repair_decode`
  call site (`model.space.apply(..., reason=[])`) resolves.
- `src/slm_training/harnesses/experiments/slm308_distance_oracle.py`, ported
  near-verbatim from fork commit `ae5448c5` — its four SLM-299 dependencies
  (`_canonical_key`, `_check_invariants`, `_enumerate_children`,
  `_normalize_inventory`) kept identical signatures on main, so the oracle's
  reverse-BFS logic is unchanged from the fork.

**Not** ported (the SLM-317/SLM-431 harness never references them):

- SLM-308's `pairwise_progress_margin` loss.
- SLM-310's `corruption_action_distribution` reweighting knob.
- SLM-310's full 23-code reason-code taxonomy on `apply` — `repair_decode`
  discards the `reason` list contents, so only a single generic marker is
  ported.
- `scripts/run_slm308_distance_value.py`'s matched fixture experiment (SLM-
  308's own promotion evidence, independent of this port).

## Regression evidence

The full pre-existing `tests/test_models/test_tree_edit_diffusion.py` (10),
`tests/test_harnesses/experiments/test_slm299_edit_reachability.py`, and
`tests/test_harnesses/experiments/test_slm317_repair_hybrid.py` suites pass
**unmodified** after the port — proof the default (`mutation_count` +
`legacy`) path is byte-for-byte unchanged. 8 new tests were added to
`test_tree_edit_diffusion.py` (default-off assertions, invalid-mode
`ValueError`s, the `apply(reason=...)` accept-only shim, and a duplicate-STOP
accounting test distinguishing `legacy` vs `corrected`) plus 7 new tests in
`test_slm308_distance_oracle.py` (EXACT/BOUNDED/UNKNOWN classification,
cache identity, effective-distance edge cases).

## Powered rerun (SLM-431's preregistered rule, now executable)

```
python -m scripts.run_slm317_repair_hybrid --seeds 2..21 --min-pass-rate 0.5
```

20 seeds, disjoint from SLM-317's original `[0, 1]`, wall time ≈ 50s (well
inside the 3-minute `MAX_RUN_MINUTES` cap).

- **Safety**: PASS (`invalid_over_valid = 0`).
- **Reachability**: PASS.
- **Value**: FAIL — Wilson 95% CI `[0.0, 0.161]` over 20 seeds vs the
  preregistered `min_pass_rate = 0.5`.
- **Disposition: `repair_negative`.**

Both `ar_only` and every repair arm sit at 100% hard-valid rate on the
frozen SLM-155 fixture corpus — there is no invalid AR baseline left for
repair to fix, so no seed can show a paired improvement. This is a
legitimate, expected `repair_negative` outcome, not reframed as
inconclusive. Full evidence:
`docs/design/iter-slm317-repair-hybrid-powered-rerun-20260727.{json,md}`.

**LAR3 stays CLOSED.** SLM-421 already satisfied the recurrence-health
reopening condition; this result does not satisfy the second (do-no-harm
repair value) condition, so LAR3 is not reopened by either issue's evidence.
`SLM-432` (LAR4-06) remains correctly gated: it requires LAR3 authorization
jointly on both conditions, which is not met.

## Non-goals honored

- No change to the do-no-harm commit rule, the frozen SLM-155 corpus, or any
  ship/production claim.
- No weakening of the preregistered `min_pass_rate = 0.5` threshold to force
  a decision.
- No change to the frozen SLM-279 depth-supervision arithmetic contract
  (unrelated model/objective).
- LAR3 is not declared reopened or closed beyond what the powered rerun
  result itself states.
