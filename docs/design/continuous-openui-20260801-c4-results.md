# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c4` |
| Predecessor | `continuous-loop-20260801-c3` |
| Role / intent | `promotion` / `confirm` (champion-queue confirmatory retest) |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 42 / batch 2 / seed 100004 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c4-control | bounds off, canvas off | 3 | 1.0 | 0.333 | 11518.24 | eval completed; ship gates fail (insufficient n only) |
| c20260801-c4-confirm | bounds **on**, canvas **on** | 3 | 1.0 | 0.333 | 12036.38 | eval completed; ship gates fail (same) |

Primary delta (confirm − control) p50 latency: **+518.14 ms** (positive =
candidate slower) — the sign flipped versus cycle 3's screening result
(-146.62 ms at seed 100003).

## Champion queue outcome

Confirmatory retest of `champ-continuous-openui-20260801-3-3ef2d4724c8df79e`
(bounds+canvas, `steps=42`), same levers, new seed:

```text
status: queued -> rejected
confirm_attempts: 1
confirm_campaign_id: continuous-loop-20260801-c4
```

**This is the champion-queue design working as intended.** The cycle-3
screening win did not reproduce at a new seed — it was seed variance at
`n=3`, not a genuine lever effect. Rejecting here is correct; a stack layer
would have been wrong.

## SDLC Phase A classification

Non-positive (`primary_metric_unavailable` for the promotion-role primary
`held_out.structural_similarity` — no `held_out` suite at this fixture
scale — plus `fixture_insufficient_n` on both arms). No stacked PR opened.
Local commit only.

## Diagnostics

1. Absolute latency for both arms this cycle (~11.5-12.0s) is roughly 3x
   cycle 3's arms (~3.8-3.9s) at the identical `steps=42` recipe — seed
   100004 is producing longer/more elaborate decode output than seed 100003
   before hitting the same grammar/timeout ceiling. This reinforces that
   single-seed, `n=3` latency deltas are not reliable signal; the champion
   queue's confirm-then-promote gate is the correct guard against exactly
   this.
2. `bounds`-only and `canvas`-only were still not isolated (proposed as
   `c20260801-c4-bounds` / `-canvas` at `steps=1042/1043` but not executed
   this cycle — budget capped at control + 1 candidate).

## Next-run priorities

1. **model:** do not re-propose the rejected bounds+canvas combo as a
   champion without first isolating `bounds`-only vs `canvas`-only effects.
2. **evaluation:** single-seed `n=3` latency comparisons are noisy enough to
   flip sign cycle-to-cycle; treat screening wins as leads, not conclusions,
   until confirmed (which is already this repo's policy — this cycle is the
   evidence for why).
3. **loop:** continue with fresh hypotheses next cycle; no open champion
   remains queued.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/` (local, not tracked)
- Runs: `.../runs/c20260801-c4-control/`, `.../runs/c20260801-c4-confirm/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c4/sdlc_delivery.json`
- Champion queue: `outputs/autoresearch/loops/continuous-openui-20260801/champion_queue.jsonl` (local, not tracked)
- Predecessor: `continuous-openui-20260801-c3-results.md`
