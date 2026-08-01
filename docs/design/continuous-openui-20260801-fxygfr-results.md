# Continuous autotrain loop (scheduled session, branch fxygfr) results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Source | `8a76f949796444d87bdf5369933e40fc89eebdc7` (origin/main) |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` / suite `smoke` |
| Wall cap | 3 minutes per campaign |

## Environment setup (fresh scheduled session)

Same footgun as prior scheduled sessions: no venv, no node_modules on
container start.

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
NODE_OPTIONS="--max-old-space-size=8192" npm ci
```

This session's `NODE_OPTIONS` is malformed
(`"--import tsx" --max-old-space-size=8192`, literal quotes, `--import` not
permitted in `NODE_OPTIONS` at all) and crashes the AgentV runner subprocess.
The fix already exists, unmerged, in **PR #1264** (`claude/great-dirac-37mit6`
→ `src/slm_training/evals/agentv.py`). Not duplicated here; worked around by
overriding `NODE_OPTIONS` on every subprocess invocation, matching the
approach documented by the immediately-prior scheduled session
(`continuous-openui-20260801-g7f4y70-results.md`, unmerged on
`claude/great-dirac-7f4y70`).

## Cycle 2: canvas lever screening (soft failure -- host-load timeout)

| Arm | Levers | Result |
| --- | --- | --- |
| c2-control | bounds off, canvas off | eval completed; **degenerate** -- all 3 smoke docs hit the 24s decode timeout (`decode_timeout_rate=1.0`), no parses |
| c2-canvas | canvas on | `CYCLE_ERROR TimeoutExpired` at 74.7s -- ran out of the campaign's shared 3-minute wall budget after the control arm's full-timeout decode consumed most of it |

No `cycle_handoff.json` / `sdlc_delivery.json` was written for cycle 2 (the
driver errored before reaching that stage). Per the continuous-mode absolute
loop law, a single timeout is a **soft failure** -- it does not stop the loop
and is not evidence of anything beyond this cycle's host load. Self-healed by
re-running the identical recipe as cycle 3.

## Cycle 3: canvas+bounds combined ("both") thrash arm vs control

| Arm | Levers | smoke n | parse_rate | mpr | structural_similarity | binder_reference_f1 | latency_ms_p50 | Ship gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | bounds off, canvas off | 3 | 1.0 | 0.0 | 0.1725 | 0.72222 | 10689.93 | **fail** (insufficient n + quality) |
| c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 0.1725 | 0.72222 | 10533.27 | **fail** (insufficient n + quality) |

Primary metric (`smoke.binder_reference_f1`, control vs. candidate):
**delta 0.0** -- exactly null, not a regression, not an improvement.

Cycle 3 completed cleanly (both arms produced real decodes, ~10.5-10.7s p50,
well under the 24s timeout) -- a large swing from cycle 2's degenerate
all-timeout control arm using the identical recipe, seed, and commit.
Consistent with prior sessions' repeatedly-documented host-load variance in
this sandbox (see `continuous-openui-20260801-g7f4y70-results.md` cycle 3's
own ~3x wall-clock swing note); not investigated further as a lever effect.

**SDLC Phase A classification:** `NON_POSITIVE`, `stack_layer=false`.
Reasons: `fixture_insufficient_n:c20260801-c3-both`,
`fixture_insufficient_n:c20260801-c3-control`,
`primary_metric_null_or_worse:smoke.binder_reference_f1:control=0.72222
candidate=0.72222 improvement=0.0`, `fixture_insufficient_n_alone`.

No harness signals were raised (`harness_signals: []` in both cycle traces) --
the `held_out`/`adversarial`/`ood`/`rico_held` "missing_suite" gate lines seen
in raw output are expected: this cycle's `eval_suites` knob requested `smoke`
only (a fast screening pass), matching the pattern of prior screening cycles
in this loop. `eval_version` was correctly bound to
`e938_role_safe_all_targets_v2` throughout (no v1-default footgun this
session).

## Next-run priorities

1. **model:** the hypothesis matrix this cycle also proposed `c3-steps`
   (steps 20→40) and `c3-batch1` (batch_size 2→1) arms that were not selected
   under the 1-experiment-pair campaign budget -- run those next instead of
   repeating another canvas/bounds screen (canvas and bounds have now been
   screened, individually and combined, across at least 4 prior sessions
   today with consistently null deltas).
2. **infrastructure:** do not chase the cycle-2 decode-timeout arm further as
   a regression -- it did not reproduce in cycle 3 under the identical
   recipe/seed/commit; single-cycle host-load variance per continuous-mode
   soft-failure policy.
3. **evaluation:** keep ship gates honest; `smoke` suite is fixed at `n=3`
   (need >=20) so `insufficient_n` fails by construction at this fixture
   scale -- expected, not a blocker.

## Open-PR-stack observation (not part of this cycle's classification)

At session start, `tyler-r-kendrick/slm-training` had **~47 open pull
requests**, the large majority `docs(autotrain): continuous loop ...
(non-positive, docs only)` from repeated scheduled sessions across
2026-07-27 through 2026-08-01, none yet closed out. The continuous-mode
contract's "when training stops" SDLC Phase B bottom-up closeout has
apparently not run in several days of scheduled firings. Flagging for the
user's attention; out of scope for this cycle to close out unilaterally.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c{2,3}/`
  (gitignored, not committed)
- JSON twin: `continuous-openui-20260801-fxygfr-results.json`
