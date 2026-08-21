# Hill-climb recovery integrator (2026-08-21)

Fixture-scale harness change, **not** a ship claim. Live loop
`continuous-openui-local` was not killed; this documents swarm merge.

## Recipe

- Device: loop host CPU (CUDA routing added, unused unless present)
- `MAX_RUN_MINUTES`: 3 (unchanged)
- Honesty: fixture diagnostic
- Policy: `autotrain_climb` **v14**
- Components: `harness.autoresearch.experiment_campaign` v264,
  `autoresearch.conclusions` v6, `data.test_build` v8, `model.twotower` v319

## Merged swarms

| Swarm | Outcome |
| --- | --- |
| METRIC | Screening primary `smoke.eval_nll` decrease; SS secondary |
| BUDGET | Steps fitted to train floor; CUDA auto |
| CHAMPION | `loops/<id>/champion/` warm-start; confirm-only advance |
| STATS | Wilcoxon + `power_floor_n`; smoke24 published |
| DATA | `hillclimb_strict_v1` 976 records (strict, not 1000) |
| MULTIARM | k candidates vs shared control; k fits stage wall |
| DECODE | Domain-restricted rank; train-only n-gram regen |
| OPS | Compose heal only when bank empty; Serena YAML restore |

## Interface patches (this commit)

- `defaults.train_version` = `hillclimb_strict_v1`
- `promotion_suite_n` = 24
- `published_smoke` prefers `e938_role_safe_all_targets_smoke24_v1`

Three consecutive scratch cycles were **not** run here: the live supervisor
owns the driver lock. First post-merge live cycles must show NLL primary,
fitted steps, and no `thrash_bank_compose` on an already-open bank.
