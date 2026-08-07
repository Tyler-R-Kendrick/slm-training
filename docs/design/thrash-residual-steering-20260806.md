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

## Artifacts

| Path | Role |
| --- | --- |
| `loops/<id>/interesting_residuals.jsonl` | Append-only residual observations |
| `loops/<id>/slug_stats.json` | Regenerable prior (via miner) |
| `THRASH_RESIDUAL` / `THRASH_SOFT_RANK` logs | Driver visibility |

## Tools

```bash
python -m scripts.mine_continuous_residuals \
  --root outputs/autoresearch --loop-id continuous-openui-local --write-ledger
```

Pure helpers: `src/slm_training/autoresearch/thrash_residuals.py`.

## Non-goals

- Weakening ship gates or fixture honesty  
- Auto-promote from residuals  
- LLM hypothesizer as thrash controller  
- Full retrain densification (optional later: eval-only confirm-lite)

## Related

- Dual regime thrash: `docs/design/dual-regime-thrash-20260804.md`
- Thrash timing Pareto: `docs/design/autotrain-thrash-timing-pareto-20260803.md`
