# G2 — recipe-evolution harness fixture campaign (2026-07-17)

Fixture-grade wiring run for Track G2 (Linear SLM-35). Machine-readable
evidence:
[recipe-evolution-results-iter-g2-20260717.json](recipe-evolution-results-iter-g2-20260717.json)
(mirror of the campaign's `population.json`). Code:
[`src/slm_training/harnesses/experiments/recipe_evolution.py`](../../src/slm_training/harnesses/experiments/recipe_evolution.py)
+ [`scripts/run_recipe_evolution.py`](../../scripts/run_recipe_evolution.py).

## What was built

AlphaEvolve/ShinkaEvolve-**Adapted** (only the population/evaluator pattern
transfers — no LLM-guided program mutation, no evolved code; the evaluator
is the existing frozen train/eval/gate stack):

- **Bounded typed gene space** over recipe knobs only (corruption:
  `mask_pattern`, `statement_mask_prob`, `remask_span`; decode: `gen_steps`,
  `parallel_unmask`; loss: `ltr_loss_weight`, `fidelity_loss_weight`;
  optimizer: `lr`). Model geometry is deliberately not a gene. Every
  mutation re-validates against the domain (`RecipeGene.__post_init__`).
- **Seeded mutation/crossover** with a unique-gene evaluation cache: each
  recipe trains exactly once per campaign (the autoresearch
  no-repeated-knob-signature contract; elites are cached, never re-trained).
- **Gate-locked selection**: `rank_candidates` has no threshold knob —
  candidates passing the unaltered `DEFAULT_SHIP_GATES` strictly outrank
  every gate-failing candidate, missing suites fail closed, and
  `promotable` is true only if the best candidate actually passed the
  frozen gates. No RL path exists; any future RL leg must sit behind
  `autoresearch.rl_gate.assert_rl_ready`.
- **Persistence**: atomic local JSON campaign tree (`campaign.json`,
  `generation_*.json`, `population.json`) under `outputs/experiments/`,
  compatible with the optional, dry-by-default `sync_campaign` HF mirror.

## Fixture campaign (wiring evidence only)

Recipe: population 3, 2 generations, elite-2, 20 CPU steps per candidate,
fixture v1 corpus, fitness suites smoke+held_out, seed 0
(`g2_fixture_20260717`). Four unique recipes trained (the two elites were
cached in generation 1, as designed).

| gene | mutated knob | fitness (weighted NLL ↓) | gates |
| --- | --- | --- | --- |
| `708a9ff9e9` | `fidelity_loss_weight=0.04` | **16.415** | fail |
| `d8c96d7bae` | base recipe (`fid=0.5`) | 16.654 | fail |
| `e6b9d5f805` | decode-only knob changed | 16.654 | fail |
| `33f4de5b50` | `fidelity_loss_weight=0.76` | 16.750 | fail |

`promotable: false` — no candidate passed the frozen gates (partial fitness
suites alone guarantee `missing_suite` failures, by construction). The loop
selected, persisted, and reported honestly.

## Two instructive observations

1. **The gate lock is load-bearing, immediately.** Fitness decreased
   monotonically as mutation drove `fidelity_loss_weight` down — pure-NLL
   fitness would evolve the fidelity loss to zero. That is precisely the
   emptier-but-cheaper failure mode the honest gates guard against (the
   frozen policy includes `placeholder_fidelity` floors), and why selection
   ranks gate-passers first instead of fitness-first. At real budgets the
   fitness signal should incorporate gate-facing metrics, not replace them.
2. **Decode-only genes are fitness-neutral under NLL fitness** (`e6b9d5f805`
   ties its parent exactly): decode knobs only matter through suite metrics,
   so campaigns that evolve decode policy must select on scoreboard metrics,
   not training NLL. Recorded as a design note for the first real campaign.

## Verification

- 8 pure-Python tests (`tests/test_harnesses/experiments/test_recipe_evolution.py`):
  gene bounds + mutation determinism, `ModelBuildConfig` mapping, gate-first
  ranking (a gate-passer with worse fitness outranks a better-fitness
  gate-failer), fail-closed gate check on missing suites/metrics, campaign
  tree persistence + rerun determinism, unique-gene evaluation, and
  promotable-only-on-gate-pass.
- Fixture campaign ran end-to-end (evolve → scratch-train → eval →
  gate-checked select → persist) with zero candidate errors;
  `repo_policy`, `ruff`, `git diff --check` clean.

## Honesty and limits

- Wiring evidence only: 20-step CPU training, 4 recipes, 2 generations, one
  seed — the campaign demonstrates the loop, not a recipe improvement. No
  gate weakened, nothing promoted, no checkpoint kept.
- Fitness = `best_weighted_nll` is a placeholder selection signal; per
  observation 1 it is adversarial to fidelity at the margin and must be
  paired with (never substituted for) the frozen gates. The first real
  campaign should rank eligible candidates on gate-facing scoreboard
  metrics or `efficiency_gain_lcb`.
- The feedback-ID / predecessor-naming contract from
  `autoresearch-autotraining.md` applies when G2 campaigns feed the
  hypothesizer loop (SLM-46/G1); this harness stores unique gene signatures
  but does not yet emit typed `HypothesisFeedback` objects — that wiring is
  G1's scope.
