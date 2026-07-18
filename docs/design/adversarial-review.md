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
2. **Smoke ≈ memorization.** `src/slm_training/resources/train_seeds.jsonl` `meta.layout=smoke_align` duplicates smoke OpenUI trees (namespace-only diffs). Upsampled paraphrases amplify them. Structural similarity 1.0 + fidelity 0.0 is the tell.
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

| Suite | meaningful program | structural | component recall | placeholder_fidelity | reward |
| --- | --- | --- | --- | --- | --- |
| smoke | ≥ 0.66 | ≥ 0.35 | ≥ 0.35 | ≥ 0.25 | ≥ 0.30 |
| held_out | ≥ 0.40 | ≥ 0.30 | ≥ 0.30 | ≥ 0.15 | — |
| adversarial | ≥ 0.25 | ≥ 0.25 | ≥ 0.20 | — | — |
| ood | ≥ 0.25 | ≥ 0.25 | ≥ 0.20 | — | — |
| rico_held | ≥ 0.10 | ≥ 0.20 | ≥ 0.15 | — | — |

Smoke is a **canary**, not proof of generalization. Ship pass requires held_out + adversarial + ood + rico_held bars as well.

`component_type_recall` is the **semantic-density floor** (E2): the fraction of
the gold's component types the prediction recovers. It collapses toward 0 for
the trivial/empty program, so a compression- or decode-driven change cannot
green these gates with shorter-but-emptier output on syntax alone. The floors
sit at or below the structural bars and only make the policy stricter.

`parse_rate` now means syntactic OpenUI parse and is reported separately.
`meaningful_program_rate` is the learned-quality gate above; historical
scoreboards that predate the split used `parse_rate` for that heuristic.

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

## V6 honesty follow-ups

- `GrammarDiffusionModel.generate` no longer reads `gold.placeholders` when
  `honest_slot_contract=True` (default).
- TwoTower `generate_batch_requests` surfaces `GenerationRequest.slot_contract`
  into the prompt (`ensure_prompt_inventory`) under honest mode — restores the
  E35 inventory-in-prompt API for the production eval path.
- Prefer `--matrix v6 --only E53` (or E35/E36) for honest fixture ship claims;
  production still requires full `rico_held` (1500) + HF context.
- E292 re-evaluates the matched choice checkpoint with prompt-derived
  slot-contract constrained decoding, no DESIGN.md context, and no
  unconstrained fallback. Fidelity rises on four small suites, but meaningful
  rate remains 0.0 everywhere, component recall is at most 0.04, AgentV is 0/5,
  and 15 gates fail. This is honest fixture-scale diagnostic evidence, not a
  ship claim; see
  [iter-e292-choice-loss-suite-completeness-20260717.md](iter-e292-choice-loss-suite-completeness-20260717.md).
- E293 adds the same honest policy around a choice-native component-plan arm.
  A provenance audit found E292 was trained with DESIGN context despite a
  mislabeled summary; that reporting path is fixed. The matched DESIGN-context
  plan arm reaches one meaningful adversarial row only with bias off. In the
  no-DESIGN follow-up, the learned bias changes 38 legal choices and
  reduces gate failures 17→13 versus bias off, but meaningful rate remains 0.0
  on every suite and AgentV remains 0/5. Neither arm is a ship candidate; see
  [iter-e293-choice-component-plan-20260717.md](iter-e293-choice-component-plan-20260717.md).
- E294 supplies the separately trained no-DESIGN/no-plan control. It exactly
  matches E293 with decode bias off, so plan training alone does not improve
  discrete outputs; the active head only cuts failures 17→13 on secondary
  metrics. Meaningful rate remains 0.0, so no ship claim follows.
- E295 deterministically drops DESIGN context for exactly 240/480 training
  records. Its complete NLL lies between the all/no-DESIGN controls and frozen
  prompt-only evaluation recovers one meaningful adversarial program (0.25,
  AgentV 1/5, 14 failures), while the other four suites exactly match E294.
  This narrow signal warrants replication but does not support promotion.

## Re-eval commands

```bash
python -m scripts.build_train_data --source all --version v1 --synthesizer quality
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500

python -m scripts.evaluate_model \
  --suites smoke,held_out,adversarial,ood,rico_held \
  --train-dir outputs/data/train/v1 \
  --test-dir outputs/data/eval/v1 \
  --run-id twotower_v1_ship \
  --ship-gates
```

## 2026-07-18 — measurement-honesty remediation (policy delta)

Full audit + fixes: `measurement-honesty-remediation-20260718.md`. Gate-policy
changes (all strictly tightening; thresholds in `DEFAULT_SHIP_GATES` unchanged):

- **Evidence floor.** Every suite gate now also requires `n >= min_n`
  (`DEFAULT_MIN_SUITE_N = 20`, per-suite `min_n` override in a custom policy).
  Fixture-scale suites (n=3-5) quantize rates to k/n — smoke meaningful could
  never exceed 0.6667 — and no longer read as gateable evidence
  (`<suite>:insufficient_n`).
- **certified_fallback fails closed when unmeasured.** `fallback_count` was
  hardcoded 0 in the evaluator, so the gate was vacuously green. It is now
  summed from real decode telemetry (unconstrained retries, compiler/seeded
  fallbacks, template fallbacks) and a board without decode stats fails
  `certified_fallback unmeasured`.
- **Undefined ≠ perfect ≠ zero.** Empty-set metrics (contract precision/recall,
  placeholder fidelity/validity, component type recall) return null instead of
  a vacuous 1.0; unmeasured aggregates are null, never a fabricated 0.0; the
  existing missing-metric-fails rule turns null into an honest gate failure.
  `metric_defined_n` discloses every mean's denominator.
- **Decoder-guaranteed labeling.** Payloads list metrics enforced by the active
  decode policy (`decoder_guaranteed`) — post-E226 `parse_rate` is
  grammar/compiler-guaranteed and must not be read as model skill. Readers only
  substitute `parse_rate` for the meaningful lever on pre-split boards (both
  meaningful and `syntax_parse_rate` absent) and tag it
  `meaningful_source=parse_rate_legacy` (rendered with `*`).
- **Promotion unblocked from a phantom metric.** `lineage` HARD_METRICS and
  `model_cycle` required `request_coverage`, which no evaluator has emitted
  since the rename — promotion now reads `contract_recall`, and a key-contract
  test pins every consumer-required metric to the evaluator's real output.
