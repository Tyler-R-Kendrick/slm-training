# Continuous autotrain loop g7f4y70 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Source | `41d874c76b9ed68f4c6d375366ea4398b95a0429` (origin/main) |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes per run |

## Environment setup (fresh scheduled session, not a code bug)

This container starts with no Python package install and no Node deps.
Getting the first cycle running required:

```bash
python3.12 -m venv .venv-autotrain && pip install -e ".[dev]"
npm ci   # repo root, for node_modules/@agentv/core
```

**Cycle 1** still failed closed with `RuntimeError: AgentV SDK is
unavailable`. Root cause: this sandbox's session-level `NODE_OPTIONS` is a
malformed `"--import tsx" --max-old-space-size=8192` (literal quotes
included), which crashes the AgentV runner subprocess with exit 9 —
**already found and fixed in [PR #1264](https://github.com/Tyler-R-Kendrick/slm-training/pull/1264)**
(open, unmerged as of this session, branch `claude/great-dirac-37mit6`,
sanitizes `NODE_OPTIONS` in `src/slm_training/evals/agentv.py`). Since that
fix is already in flight from a concurrent session, this session did **not**
duplicate it — it worked around the same symptom by overriding
`NODE_OPTIONS` on the shell command line for every subsequent invocation.
Once #1264 merges, this loop-id should rebase and drop the manual override.

## Cycle 2: canvas lever screening

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Ship gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | `compact_active_canvas=off` | 3 | 1.0 | 0.0 | 0.28 | 22467.2 | **fail** (insufficient n + quality) |
| c2-canvas | `compact_active_canvas=on` | 3 | 1.0 | 0.0 | 0.28 | 22147.61 | **fail** (insufficient n + quality) |

Primary metric delta (control − canvas) p50 latency: **319.59 ms**
improvement, but `mpr=0.0 < 0.333` — rejected as a pure latency blip per
`_classify_metric_tradeoff` (latency wins require held mpr ≥ ~1/3).

**Classification:** NON_POSITIVE (`fixture_insufficient_n`,
`latency_win_rejected_low_mpr`).

## Cycle 3: grammar-completion-bounds lever screening

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Ship gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | `grammar_completion_bounds=off` | 3 | 1.0 | 0.0 | 0.1725 | 7787.5 | **fail** (insufficient n + quality) |
| c3-both | `grammar_completion_bounds=on` | 3 | 1.0 | 0.0 | 0.1725 | 7944.33 | **fail** (insufficient n + quality) |

Primary metric delta (control − bounds): **−156.83 ms** (worse). Wall-clock
for both arms dropped ~3x versus cycle 2 (22.1–22.5s → 7.8–7.9s p50) despite
an identical recipe; not investigated further this session (likely host-load
variance across the two invocations, not a lever effect — flagged as an
open question rather than a claim).

**Classification:** NON_POSITIVE (`primary_metric_null_or_worse`,
`fixture_insufficient_n_alone`).

## Cycle 4: promotion-intent screening

| Arm | smoke mpr | smoke structural_similarity | held_out n | held_out structural_similarity |
| --- | ---: | ---: | ---: | ---: |
| c4-control | 0.5 | 0.3625 | 5 | 0.38248 |
| c4-steps | 0.333 | 0.51 | 5 | 0.37006 |

Primary metric delta (steps − control) on `held_out.structural_similarity`:
**−0.01242** (worse). Note the smoke-suite and held-out-suite structural
similarity move in opposite directions between arms (steps arm scores
higher on smoke, lower on held-out) — a reminder that a 3-document smoke
suite is not a reliable proxy for the 5-document held-out suite at this
fixture scale, consistent with why both suites independently fail
`insufficient_n`.

**Classification:** NON_POSITIVE (`primary_metric_null_or_worse`,
`fixture_insufficient_n_alone`).

## Next-run priorities

1. **infrastructure:** rebase this loop-id onto PR #1264 once merged so
   future cycles don't need the manual `NODE_OPTIONS` override.
2. **harness:** the published `smoke` suite is fixed at `n=3` (need ≥20), so
   `insufficient_n` fails on every screening cycle by construction — this is
   an expected fixture-scale diagnostic per the repo's own honesty rules,
   not a blocker to chase further from this loop.
3. **model:** per `autotrain-loop-ledger-20260725.md`'s own diminishing-
   returns notes, single-variable `wf_smoke_v2` lever sweeps (steps, seed,
   batch-size, canvas, bounds) are now covered across many prior sessions.
   The next iteration of this loop-id should pick up an actually-open thread
   (DSH5-10 preference-training scope, or the next queued `AP-007+`
   campaign arm) instead of another identical smoke-fixture screening pass.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c{1,2,3,4}/`
  (gitignored, not committed)
- JSON twin: `continuous-openui-20260801-g7f4y70-results.json`
