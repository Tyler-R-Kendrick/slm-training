# Thrash residual steering (cheap, no retrain)

**Honesty:** harness design. Not a ship-quality model claim.

## Problem

Continuous thrash already writes rich `sdlc_delivery.json` evidence then mostly
**throws away interesting residuals** (e.g. high structural similarity with
binder non-regression fail). Trajectory falls back to cycle-number bank
rotation. Full retrain densification is too expensive for every spike.

## Decision

Mine completed deliveries (JSON-only) into a loop-local residual ledger and
**soft-rank** open thrash slugs. Interesting residual ≠ promotable.

| Residual class | Trigger | Steering |
| --- | --- | --- |
| `primary_up_binder_down` | primary lift / efficiency + binder F1 non-regression fail | Soft-boost that slug / family |
| `efficiency_win_quality_held` | efficiency_win + quality_held | Soft-boost |
| `high_band_absolute` | max SS ≥ 0.35 with primary lift | Soft-boost |
| `control_spike_shared` | both arms high SS, near-tie | **No densify** (fixture draw) |

Multi-seed arm close and hard `skip` still outrank soft boosts.

**Family prior:** `primary_up_binder_down` also lightly boosts binder-focused
bank slugs (`BINDER_FAMILY_SLUGS`). Compose residuals share a light sticky bump
across observed compose-* residual slugs.

## Artifacts

| Path | Role |
| --- | --- |
| `loops/<id>/interesting_residuals.jsonl` | Append-only residual observations |
| `loops/<id>/slug_stats.json` | Regenerable prior (miner or post-cycle refresh) |
| `loops/<id>/residual_eval_queue.jsonl` | Optional eval-only confirm-lite notes + checkpoint paths |
| `THRASH_RESIDUAL` / `THRASH_SOFT_RANK` logs | Driver visibility |

Driver: after each complete delivery, classify → append residual → refresh
`slug_stats` (last ~120 deliveries) → queue eval-lite note if checkpoint exists.
Soft-rank open thrash slugs on the next screening selection.

## Tools

```bash
python -m scripts.mine_continuous_residuals \
  --root outputs/autoresearch --loop-id continuous-openui-local --write-ledger
```

Pure helpers: `src/slm_training/autoresearch/thrash_residuals.py`.

## Eval-only confirm-lite (Layer 2, manual / queued)

When a residual has a valid `checkpoints/last.pt`, the driver appends
`residual_eval_queue.jsonl` (`status=queued`). This does **not** auto-run eval
(wall budget + lineage fail-closed). Operators/agents may:

```bash
python -m scripts.evaluate_model \
  --checkpoint <path-from-queue> \
  --test-dir <eval_version> --suites smoke --ship-gates ...
```

Auto eval-only as a continuous cycle role remains deferred (needs wall-cap
scheduling + replay digests). Soft re-rank is the automatic path.

## Non-goals

- Weakening ship gates or fixture honesty  
- Auto-promote from residuals  
- LLM hypothesizer as thrash controller  
- Full retrain densification

## Related

- Dual regime thrash: `docs/design/dual-regime-thrash-20260804.md`
- Thrash timing Pareto: `docs/design/autotrain-thrash-timing-pareto-20260803.md`
