# Autotrain c1743: component-plan is a quality-held latency win

**Verdict:** the `component-plan` arm exactly ties the matched control on
every quality metric (parse, meaningful-program, structural similarity,
binder F1, reward) on this fixture, and is 5.79% faster on completed-document
p50 latency (2,539.57 → 2,392.58 ms). The driver's own SDLC Phase A
classifier calls this **positive**: an efficiency win (`mpr_per_ms` gain
6.14%, above the 5% policy floor) with quality held. This is the first
positive result in this session's continuous run, after two prior
`grammar_completion_bounds` cycles ([c1741](autotrain-cycle-1741-grammar-bounds-latency-regression.md),
[c1742](autotrain-cycle-1742-grammar-bounds-repeat-screen.md)) were rejected.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | Reward | p50 (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,755,764 | 3/3 | 0 | 1.000 | 0.3333 | 0.23083 | 0.4889 | 0.5493 | 2,539.57 | fail (fixture n=3) |
| component-plan | 1,755,764 | 3/3 | 0 | 1.000 | 0.3333 | 0.23083 | 0.4889 | 0.5493 | 2,392.58 | fail (fixture n=3) |

Both arms complete the full 3-record smoke suite with zero decode timeouts —
a complete, scoreable comparison. Ship gates fail on both for the expected
fixture-scale reason (`smoke:insufficient_n`, n=3 vs the ≥20 floor); this is
a screening-level efficiency signal, **not** a promotion or ship claim.

## Signals and next run

- `component-plan` is quality-held (exact tie on every scored metric) and
  5.79% faster on completed-document p50 latency; the efficiency-win test
  (`mpr_per_ms` gain 6.14% ≥ 5% floor) passes.
- Per `sdlc` autotrain-iteration-delivery, a positive result with tracked
  docs delta gets a stacked layer: this cycle's commit is pushed and a PR is
  opened for `claude/great-dirac-tb5qfu` (see the cycle_handoff
  `deliver_stack` action).
- This is still fixture-scale (n=3, single seed) screening evidence — it
  queues a champion candidate (`CHAMPION_ENQUEUE`), it does not promote or
  unblock ship gates. A confirmatory run with a held-out/larger-n suite is
  needed before any promotion claim.
- Both checkpoints are local fixture-scratch artifacts (`outputs/`,
  gitignored, no sync) — neither reusable, promotable, nor ship at this
  evidence tier.
- Next: continue the loop; the driver's ranked priorities favor rotating the
  lever bank next rather than re-running `component-plan` immediately.

Machine-readable evidence is in
[`autotrain-cycle-1743-component-plan-latency-win.json`](autotrain-cycle-1743-component-plan-latency-win.json).
