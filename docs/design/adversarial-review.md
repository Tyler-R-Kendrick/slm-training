# Adversarial review — OpenUI TwoTower ship cycle

Rubber-duck / red-team audit of what shipped under `twotower_v1_ship` and the surrounding harnesses. **Verdict: the run is a fixture memorizer with soft gates, not a generalizable ship.** Treat existing `gates.json` `"pass": true` as invalidated.

## Bottom line

| Claim (docs / gates) | Reality |
| --- | --- |
| Ship gates pass | Soft policy: only smoke (+ weak held_out) counted; hard suites ignored |
| Smoke parse 1.0 = readiness | Near-isomorphic train trees + `smoke_align` seeds; fidelity **0.0** |
| DESIGN.md conditioning | Ship train used `design_md_in_context=false`, scratch context |
| Large RICO / HF ship | Eval scoreboard `rico_held n=23` while test corpus has **1500**; train was `v1_fixture_up` (~141) |
| `remote_train.py` works on fresh clone | Builds `v1`, trains `v1_fixture_up` (missing) |
| DPO stage | Reference-free surrogate; pairs often valid vs broken corruption |

## Findings

### P0 — correctness / honesty

1. **Gates exclude hard suites.** `evaluate_model --suites …` only fail-unders the primary (smoke). Hand-written `gates.json` marks held_out pass at parse ≥ 0.15 (1/5). `rico_held` / adversarial parse **0.0** never fail the ship.
2. **Smoke ≈ memorization.** `fixtures/train_seeds.jsonl` `meta.layout=smoke_align` duplicates smoke OpenUI trees (namespace-only diffs). Upsampled paraphrases amplify them. Structural similarity 1.0 + fidelity 0.0 is the tell.
3. **Stale rico_held eval.** Scoreboard n=23 vs manifest suite_counts `rico_held=1500`.
4. **`remote_train.py` corpus mismatch.** Builds `outputs/train_data/v1`, trains `v1_fixture_up`.
5. **Ship corpus ≠ claimed recipe.** Fixture upsample, scratch, no DESIGN.md in context — docs implied HF + DESIGN.md + broad RICO.

### P1 — metric / signal integrity

6. **`placeholder_validity` forgives train namespaces** via `_normalize_placeholder`; fidelity stays 0 and was not gated.
7. **`design_lint_score` / reward lint gold DESIGN.md**, not the prediction — inflates reward when context was disabled.
8. **Meaningful parse** accepts truncated but valid programs (e.g. settings → single `TextContent`).
9. **“DPO” has no reference model**; preference pairs often gold vs `BrokenText`-style rejects.
10. **Exact fingerprints miss structural / prompt near-duplicates** (`:hero.title` vs `:smoke.hero.title`).

### P2 — quality / process

11. RICO Card/TextContent homogeny; empty-Stack collapse; orphan soft `gates.json`; no train-time multi-suite eval loop.

## Remediation plan

| ID | Action | Status |
| --- | --- | --- |
| R1 | Document this review; separate **fixture demo** vs **ship readiness** in README / design spec | done |
| R2 | Remove `smoke_align` train seeds; reshape smoke fixtures so trees are not isomorphic to train | done |
| R3 | Structural OpenUI fingerprint in leakage (placeholder- and binder-normalized) | done |
| R4 | Honest multi-suite `write_ship_gates` + `--ship-gates`; gate fidelity, not soft validity alone | done |
| R5 | Meaningful parse requires gold component-type recall; reward/eval stop crediting gold DESIGN.md lint | done |
| R6 | Fix `remote_train.py` to train the corpus it builds; multi-suite ship gates | done |
| R7 | Rebuild train/test; re-score `twotower_v1_ship` (or note fixture-demo only); write honest `gates.json` | follow-up eval |
| R8 | Preference stage: label reference-free; prefer grammar-valid rejects over broken syntax (later) | documented |

## Honest ship gate policy

All evaluated suites must be checked. Defaults (CLI `--ship-gates`):

| Suite | parse | structural | placeholder_fidelity | reward |
| --- | --- | --- | --- | --- |
| smoke | ≥ 0.66 | ≥ 0.35 | ≥ 0.25 | ≥ 0.30 |
| held_out | ≥ 0.40 | ≥ 0.30 | ≥ 0.15 | — |
| adversarial | ≥ 0.25 | ≥ 0.25 | — | — |
| ood | ≥ 0.25 | ≥ 0.25 | — | — |
| rico_held | ≥ 0.10 | ≥ 0.20 | — | — |

Smoke is a **canary**, not proof of generalization. Ship pass requires held_out + adversarial + ood + rico_held bars as well.

## Fixture demo vs ship

- **Fixture demo:** tiny upsample (`v1_fixture_up`), scratch context, smoke-only soft thresholds — useful for CI wiring, **not** a product claim.
- **Ship candidate:** train `v1` (all sources + quality synth), HF context + DESIGN.md in context when claimed, full scoreboard + `--ship-gates`, rico_held at full test size.

## Honest re-eval (`twotower_v1_ship`, post-remediation metrics)

Checkpoint remains the pre-remediation fixture-upsample scratch run (not retrained). After fixture reshape + honest metrics:

| Suite | n | parse | fidelity | struct | reward |
| --- | --- | --- | --- | --- | --- |
| smoke | 3 | 1.0 | **0.0** | 0.68 | 0.65 |
| held_out | 5 | **0.0** | 0.0 | 0.48 | 0.13 |
| adversarial | 4 | **0.0** | 0.0 | 0.45 | 0.0 |
| ood | 4 | 0.25 | 0.0 | 0.45 | 0.16 |
| rico_held | **1500** | **0.0** | ~0 | 0.17 | 0.0 |

`--ship-gates` → **fail** (fidelity / held_out / adversarial / rico_held). Prior `"pass": true` remains invalidated.

## Re-eval commands

```bash
python -m scripts.build_train_data --source all --version v1 --synthesizer quality
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500

python -m scripts.evaluate_model \
  --suites smoke,held_out,adversarial,ood,rico_held \
  --train-dir outputs/train_data/v1 \
  --test-dir outputs/test_data/v1 \
  --run-id twotower_v1_ship \
  --ship-gates
```
