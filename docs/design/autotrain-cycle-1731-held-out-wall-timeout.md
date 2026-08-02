# Autotrain c1731: promotion-cycle held_out wall timeout (non-positive, soft)

**Verdict:** promotion-role cycle `continuous-loop-20260802-continuous-openui-202607-98199209-c4`
requested `--suites smoke,held_out` for both the `control` and `steps` arms,
targeting primary metric `held_out.structural_similarity`. Both arms trained
successfully, but the combined `smoke,held_out` evaluation exceeded the
repo-wide 3-minute `MAX_RUN_MINUTES` wall cap on this CPU-only container and
was killed before writing a usable scoreboard for either arm.

```
stage exceeded wall-time limit:
.venv/bin/python -m scripts.evaluate_model --test-dir e938_role_safe_all_targets_v2
  --run-root .../c4/runs --run-id c20260802-continuous-openui-202607-98199209-c4-control
  --ship-gates --honest-slot-contract --slot-contract-constrained-decode
  --train-version wf_smoke_v2 --suites smoke,held_out --decode-timeout-seconds 24.0
  --local-files-only
```

`held_out` records (e.g. `held_out_form_01`, `held_out_dual_card_01`) are
longer/more complex than the 3-record `smoke` fixture, and CPU-only
constrained decode for both suites in a single process does not fit inside
the 3-minute cap on this container. This is the **first** occurrence of this
specific timeout signature — soft failure per the loop law, not a repeated
hard blocker (needs 3 identical occurrences with no new information to
qualify). The `steps` arm left a partial `eval_smoke.json` with an unverified
`structural_similarity: 0.51`; that write happened before the kill and is not
treated as a confirmed measurement here.

## SDLC Phase A classification

`SDLC_PHASE_A NON_POSITIVE` — `wall_timeout` + `empty_metrics` on both arms,
`primary_metric_unavailable`. No stack layer opened.
`cycle_role: promotion`, `climb_state: inconclusive`, `ship_state: blocked`.

## Next-run priorities

1. The queued `retry_measurement` action on this campaign replays the
   identical frozen `control`/`steps` arms automatically on the next
   supervised cycle, per the driver's frozen-replay consumption rule.
2. If the same `smoke,held_out` wall-timeout recurs on replay, that is
   occurrence 2/3 toward the hard-block threshold; a third identical
   occurrence with no new information should route to a typed
   `HarnessSignalV1` (`model_build`) via `improve-openui-harnesses` — e.g.
   evaluating `held_out` in a separate stage/process from `smoke`, or scoping
   promotion-role continuous cycles to a suite set that fits the CPU wall cap
   — rather than a fourth blind retry.
3. Screening-role cycles (c1729/c1730) are unaffected: they only request the
   `smoke` suite and complete well inside the cap.

No checkpoint was promoted; both checkpoints are unpromoted local scratch
artifacts of a timed-out promotion attempt. No model-card or README update
applies. Machine-readable evidence is in
[`autotrain-cycle-1731-held-out-wall-timeout.json`](autotrain-cycle-1731-held-out-wall-timeout.json).
