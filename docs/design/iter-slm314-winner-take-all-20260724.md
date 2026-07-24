# SLM-314 (LAR2-05): winner-take-all over multi-mode gold

**Verdict: `rejected`** (fixture-scale matched arms; not a ship claim)

## Preregistered (locked before results)

- arms: `["single_gold", "multi_gold", "wta"]` — identical model init, steps, optimizer, per-step prompt coverage, seeds
- floor epsilon: 0.1; retained iff eval-loss improvement > 0.0 vs matched baseline and within 0.5 of the prompt's best-improved mode
- rule: wta_preserves_modes iff coverage(wta) > coverage(single_gold) AND v2 semantic verdict regression vs single_gold <= 0.05, else rejected
- frozen dataset: `src/slm_training/resources/data/slm314_multimode/modes.jsonl` (8 prompts x 2 modes, manifest `10d1edc939cc74c1…`)

## Synthetic two-mode proof (deterministic)

| arm | p(mode A) | p(mode B) | p(invalid hybrid) |
| --- | --- | --- | --- |
| single_gold | 0.987 | 0.000 | 0.013 |
| multi_gold | 0.250 | 0.250 | 0.500 |
| wta | 0.826 | 0.008 | 0.165 |

Assertions: single_gold_collapses=PASS, wta_retains_loser_mode=PASS, wta_winner_dominant=PASS, multi_gold_hybrid_invalid=PASS

## Matched arms (real tree-edit model, frozen multi-mode dataset)

| arm | mode coverage | collapse | hard-valid | mode-hit | v2 verdict |
| --- | --- | --- | --- | --- | --- |
| single_gold | 0.750 | 0.250 | 1.000 | 0.125 | 0.000 |
| multi_gold | 0.500 | 0.500 | 1.000 | 0.000 | 0.000 |
| wta | 0.500 | 0.500 | 1.000 | 0.125 | 0.000 |

Coverage gain (wta − single_gold): **-0.250**; v2 semantic regression: **0.000** (budget 0.05). Multi-gold coverage: 0.500.

WTA per-step selected-mode telemetry: `outputs/slm314/wta_telemetry.jsonl`.

## Honesty

Fixture-scale: 8 prompts x 2 modes is far below any ship-gate prompt count — mechanism evidence only, not a production ship claim. Mode identity is alpha-invariant canonical AST fingerprints (never serialization strings); the frozen dataset is hash-verified before training. All arms share model init, steps, optimizer, per-step prompt coverage, and seeds. Decode is deterministic beam search (unique valid ASTs per prompt == 1 by construction); coverage is measured by deterministic per-mode eval loss instead. The SLM-130 ambiguity sets are wiring reports without a committed multi-mode corpus, so this dataset is synthesized and frozen in-repo.
