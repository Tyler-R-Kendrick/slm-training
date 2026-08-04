# Continuous autotrain: 2026-08-04 cycle 2 (`continuous-openui-local-2`) — independently reproduces the branch-point-driven decode cost on a fresh checkpoint

**Verdict:** infrastructure non-completion (decode timeout), but this time it
is **confirmatory evidence**, not a new mystery: a freshly-trained checkpoint
(not the old stale gd6j83-c2 one) hits the same ~23s/record compiler-tree
decode cost on the `component-plan` hypothesis, symmetric across control and
candidate, exactly matching the pattern
[`decode-compiler-tree-branch-point-cost-finding.md`](decode-compiler-tree-branch-point-cost-finding.md)
characterized as legitimate, bounded-per-branch-point cost rather than a bug.

## Result

| Arm | compiler_ms_mean | decode_timeout_count | n |
| --- | ---: | ---: | ---: |
| control | 23099.2 | 3/3 | 3 |
| component-plan | 23055.9 | 3/3 | 3 |

Both arms decode via a **freshly-trained** checkpoint (integration commit
`8e006b63`, no relation to the old gd6j83-c2/c3/c4 frozen manifests) and both
still time out identically. This is independent confirmation that the
~23s/record cost is a property of this recipe/hypothesis combination's
branch-point count, not an artifact of one particular stale checkpoint or a
one-off seed — the branch-point-cost finding's central claim (bounded per
point, but multiplied by however many ambiguous branch points a trajectory
visits, which can be a genuinely branch-rich `component-plan` recipe) holds
up under a completely fresh measurement.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure` on both arms).
No stack layer.

## Disposition

Per the branch-point-cost finding's own policy recommendation #2 (a
mode-aware fair-share timeout reservation tied to measured branch-point cost,
proposed but not implemented there for lack of real-checkpoint evidence):
this cycle now provides exactly that missing evidence — two independent
fresh measurements (this cycle's `component-plan`/control pair) both
landing at ~23s/record for this specific hypothesis at
`decode_timeout_seconds=8.0` (`effective` 24.0s). A future session should
use this as the sizing basis for that reservation, rather than treating
every future occurrence of this exact symmetric timeout as a fresh
investigation. This session does not implement that reservation itself
(out of scope for a screening cycle; needs its own dedicated
`improve-openui-harnesses` pass with a regression test), per the loop rule
against blind timeout changes without evidence — the evidence now exists,
but sizing and implementing the fix is separate work.

## Next priorities

1. A dedicated `improve-openui-harnesses` session should implement the
   mode-aware fair-share timeout reservation in
   `_effective_record_decode_timeout`
   (`src/slm_training/harnesses/model_build/eval_runner.py:1209`), sized from
   this cycle's and the gd6j83-c2/c3 cycles' measured `compiler_ms_mean`
   values for `compiler_decode_mode="tree"`.
2. Until then, screening cycles that hit this specific `component-plan` /
   `tree`-mode combination should expect `decode_timeout_count=3` as a
   known, reproducible, non-actionable-per-cycle outcome — document and
   move on rather than re-investigating from scratch each time.
3. Rotate to a different screening hypothesis next, since `component-plan`
   under the current fixed timeout cannot produce a scoreable measurement.

Machine evidence:
[`continuous-openui-local-2-c2-confirms-branch-point-cost.json`](continuous-openui-local-2-c2-confirms-branch-point-cost.json).
