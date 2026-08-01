# Continuous autotrain cycles 1–4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Scheduled `autotrain` continuous-loop session. Source commit
`8a76f949796444d87bdf5369933e40fc89eebdc7` (clean `origin/main`, no local
patch). JSON twin: `continuous-openui-20260801-results.json`.

Driver: `python -m scripts.run_autotrain_continuous --loop-id
continuous-openui-local --supervised --max-cycles 1 --train-version
wf_smoke_v2 --steps 20`, one bounded cycle per invocation.

## Cycle 1 (`continuous-loop-20260801-c1`) — failed, infra

| Arm | Status |
| --- | --- |
| c1-control | **failed** `ModuleNotFoundError: No module named 'torch'` |
| c1-bounds | **failed** (same) |

This scheduled session started with no prior Python environment. Repair:
created a `python3.12` venv (repo requires `>=3.12,<3.13`), `pip install
--no-deps -e .` plus the same dev deps CI installs, then the CI-pinned
`torch==2.5.1+cpu` wheel from the PyTorch CPU index. Environment-only fix; no
repository code changed.

## Cycle 2 (`continuous-loop-20260801-c2`) — failed, infra

| Arm | Status |
| --- | --- |
| c2-control | **failed** `RuntimeError: AgentV SDK is unavailable; run npm ci` |
| c2-canvas | **failed** (same) |

Root cause was two-fold: the AgentV SDK's Node package had never been
installed (`npm ci` not yet run in this checkout), and the container's
inherited `NODE_OPTIONS` env var was malformed
(`"--import tsx" --max-old-space-size=8192` — a literal embedded quote
character), which made every `node` invocation fail outright. Repair: `npm ci`
at repo root, and `env -u NODE_OPTIONS` on subsequent driver invocations.
Environment-only fix; no repository code changed.

## Cycle 3 (`continuous-loop-20260801-c3`) — completed, ship gates fail (expected)

| Field | Value |
| --- | --- |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes / stage |

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | bounds off, canvas off | 3 | 1.0 | 0.0 | 0.1725 | 0.7222 | 7950.66 | ship gates fail (insufficient n + quality) |
| c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 0.1725 | 0.7222 | 8068.42 | ship gates fail (same) |

Primary metric `smoke.binder_reference_f1` delta (both − control): **0.0**.
`grammar_completion_bounds` + `compact_active_canvas` together produced no
measurable quality change at this fixture scale and a small latency increase
(+117.76 ms p50). Ship gates fail on `insufficient_n` (need ≥20, got 3) — this
is the expected outcome for a wall-capped fixture cycle, not a promotion
signal, and does not stop the loop.

`SDLC_PHASE_A NON_POSITIVE` for all three cycles: cycles 1–2 have no tracked
metrics (infra failures, self-healed via local environment repair, not a code
change); cycle 3 has a null primary-metric delta plus fixture
`insufficient_n`. No stack layer opened for any of the three per
`autotrain-iteration-delivery` (stack only on positive results).

## Cycle 4 (`continuous-loop-20260801-c4`) — timed out, soft failure

`c4-control` (steps thrash arm) exceeded the per-stage wall budget
(`TimeoutExpired` at 172.2s against the 180s cap) and was killed by the
driver. No metrics recorded. Per `autotrain` continuous-loop law, a single
wall timeout is a soft failure that never stops the loop or requires a code
fix — treated as a data point for the next cycle's step/recipe choice, not a
blocker.

## Diagnostics

1. This scheduled container starts with neither a Python venv (torch) nor
   Node deps (AgentV) provisioned, and inherits a malformed `NODE_OPTIONS`.
   All three are session/container setup issues, not repository defects.
2. Fixture-scale `wf_smoke_v2` at 20 steps still yields `smoke.n=3`, well
   under the `insufficient_n` gate of 20 — consistent with prior continuous
   cycles (e.g. `continuous-openui-20260730-c2-results.md`).
3. Combined bounds+canvas levers were indistinguishable from control on
   `structural_similarity` / `binder_reference_f1` at this scale; only
   latency moved (slightly worse).

## Next-run priorities

1. **infrastructure:** none required in-repo; document the venv + `npm ci` +
   `NODE_OPTIONS` workaround for future scheduled sessions in this container
   profile (recorded above).
2. **model:** re-test `grammar_completion_bounds` and `compact_active_canvas`
   in isolation (not combined) once a larger step/wall budget is available so
   `smoke.n` can clear 20 and the ship-quality gates become informative.
3. **evaluation:** keep ship gates honest; fixture `insufficient_n` is not
   promotion evidence.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c1/`,
  `.../continuous-loop-20260801-c2/`, `.../continuous-loop-20260801-c3/`
  (gitignored, local only)
- JSON twin: `continuous-openui-20260801-results.json`
