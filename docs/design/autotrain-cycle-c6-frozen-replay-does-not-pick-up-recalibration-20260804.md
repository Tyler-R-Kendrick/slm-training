# Autotrain c6 (continuous-openui-local, 2026-08-04 session): frozen `retry_measurement` replays don't pick up policy recalibration — by design

**Verdict:** not a bug in the `screening_decode_timeout_seconds` 8→10
recalibration (`autotrain-thrash-timing-pareto-20260804-recalibration.md`,
commit `663e2020`); it's a property of what `retry_measurement` means. c6
(another `retry_measurement` of the same frozen c2 lineage,
`frozen_manifest_sha256=57df908a9fb3218b7014f03e5da092712c739f38f27d391fb941dd4f25cc09e2`)
timed out identically to c3/c4:

| Arm | Effective budget | Decode wall (`total_ms_sum`) |
| --- | ---: | ---: |
| control | 24.0s | 24000.7ms |
| canvas | 24.0s | 24000.2ms |

Both arms' *materialized manifest* (`outputs/.../manifests/c6-canvas.json`)
records `decode_timeout_seconds: 10.0` — the new policy value — but the
actual `evaluate_model` invocation for both arms used `8.0` (confirmed in
`runs/*/eval.json` / `eval_smoke.json`), giving the unchanged `8×3=24.0s`
effective batch budget.

## Why this is correct behavior, not a regression

`retry_measurement` means "reproduce the *identical* prior measurement,"
which by design pins the original frozen arm's config (steps, batch,
seed, decode timeout, checkpoint) rather than re-deriving it from the
live policy — otherwise it would not be a valid replay of the same
experiment (continuous.md's frozen-replay reuse contract requires "exact
steps/batch/seed/learning-rate parity"). The recalibration commit changed
`policy.v1.json`, which affects **new** experiment hypotheses generated
from the current policy; it does not and should not retroactively alter an
in-flight replay of an arm that was frozen back when the policy said `8`.

## Disposition

- No further code change. The fix is real, tested (regression test in
  `663e2020`), and will apply once the loop moves off this frozen c2/c3/c4/c6
  replay chain onto a **new** screening hypothesis/matrix.
- Filed as `repair_harness` evidence (this doc) so c6's queued
  `retry_measurement` bookkeeping can close, but this session is **not**
  issuing another replay of the same frozen arm — a 4th identical replay
  under the *same* pinned 8.0s config would reproduce the same result with
  zero new information, for the reason explained above (replay does not
  pick up new policy).
- **Next iteration should start a fresh screening hypothesis/matrix**
  (not another `retry_measurement` of this lineage) to actually exercise the
  recalibrated `10.0s` budget and get the first real
  `smoke.structural_similarity` measurement for this loop-id.

Lean is `not_applicable:screening`; climb `inconclusive`; ship `blocked`.
