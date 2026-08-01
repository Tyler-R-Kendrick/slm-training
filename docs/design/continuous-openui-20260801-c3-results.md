# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c3` |
| Predecessor | `continuous-loop-20260801-c2` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 42 / batch 2 / seed 100003 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c3-control | bounds off, canvas off | 3 | 1.0 | 0.333 | 3933.91 | eval completed; ship gates fail (insufficient n only) |
| c20260801-c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.333 | **3787.29** | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **-146.62 ms** (negative = candidate faster). Quality
held (`parse_rate` 1.0→1.0, `meaningful_program_rate` 0.333→0.333);
`mpr_per_ms` efficiency improved (8.473e-05 → 8.801e-05).

`c20260801-c3-bounds`, `-canvas`, `-steps`, `-batch1` were proposed but not
executed this cycle (budget: control + 1 candidate; hypothesizer picked the
combined `both` arm).

## SDLC Phase A classification

**Positive** — primary metric win with quality held, per gate #1 in
`autotrain-iteration-delivery.md` ("Primary metric win... under the same
wall/recipe, size-matched when comparative"). `sdlc_delivery.json` →
`positive=true`, but `stack_layer=false` / `action=positive_no_tracked_delta_skip_stack`,
because `has_tracked_delta` is computed from `git status --porcelain` at the
moment the driver finishes — before this doc pair exists, the tree was clean
(cycle 2's docs were already committed).

This repo's continuous driver has a dedicated **champion queue**
(`outputs/autoresearch/loops/continuous-openui-20260801/champion_queue.jsonl`)
for exactly this case: a first screening win is **enqueued**, not stacked
immediately.

```json
{
  "entry_id": "champ-continuous-openui-20260801-3-3ef2d4724c8df79e",
  "knobs_fingerprint": "3ef2d4724c8df79e",
  "status": "queued",
  "source_candidate_id": "c20260801-c3-both",
  "knobs": {"grammar_completion_bounds": true, "compact_active_canvas": true, "steps": 42}
}
```

**No stacked PR opened this cycle.** Per the champion-queue design (see
`scripts/run_autotrain_continuous.py` promotion/confirmatory paths), a
queued screening win only becomes stack-eligible after a same-levers,
new-seed **confirmatory retest** holds the win (`queued -> confirmed`), and
— for champions that need it — a promotion cycle under held-out suites.
Docs are committed locally now so the win is not lost; the stack layer opens
once the champion queue reports `confirmed` (or `promoted`).

## Diagnostics

1. At `steps=42` (matching cycle 2's step count), `meaningful_program_rate`
   is `0.333` here versus `1.0` in cycle 2's control — different seed
   (100003 vs 100002) changes which of the 3 smoke cases parse as
   structurally meaningful. This is expected seed variance at `n=3`, not a
   regression; it is exactly why the champion queue requires a same-levers,
   new-seed confirmatory retest before trusting the win.
2. `bounds+canvas` combined was tested directly, not `bounds`-alone or
   `canvas`-alone at this step count — cycle 2 found `canvas`-alone was a
   regression at `steps=42`, so the combined win may be `bounds`-alone
   carrying the pair. Worth isolating next.

## Next-run priorities

1. **model:** run the confirmatory retest (same `bounds+canvas` levers, new
   seed) to move the champion queue entry `queued -> confirmed`.
2. **model:** isolate `bounds`-only and `canvas`-only at `steps=42` to
   attribute the win instead of crediting the pair.
3. **delivery:** only open a stacked PR once the champion is `confirmed`
   (and promoted, if the queue requires it) — a single screening cycle is
   not sufficient evidence per this repo's champion-queue policy.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/` (local, not tracked)
- Runs: `.../runs/c20260801-c3-control/`, `.../runs/c20260801-c3-both/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c3/sdlc_delivery.json`
- Champion queue: `outputs/autoresearch/loops/continuous-openui-20260801/champion_queue.jsonl` (local, not tracked)
- Predecessor: `continuous-openui-20260801-c2-results.md`
