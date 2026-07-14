---
name: honest-ship-eval
description: Use when evaluating models, writing or interpreting ship gates, claiming readiness, changing parse/fidelity/reward metrics, or deciding fixture-demo vs production ship
---

# Honest ship evaluation

## Overview

Ship readiness is **multi-suite** and **honesty-constrained**. Smoke parse alone
never proves generalization. Silent gold-placeholder channels invalidate clears.

**Policy source:** `docs/design/adversarial-review.md` and
`write_ship_gates` / `--ship-gates` in `scripts.evaluate_model`.

## Fixture demo vs ship

| Kind | Allowed | Not a ship claim |
| --- | --- | --- |
| Fixture demo | Tiny upsample, scratch, smoke wiring, CI smoke | Product readiness |
| Honest fixture ship path | Matrix clears with inventory-in-prompt, limited `rico_held` n | Production claim |
| Production ship | Full `rico_held` (1500 when claimed), HF + DESIGN.md when claimed, full scoreboard + `--ship-gates` | — |

Always name which kind you mean in docs and PR text.

## Gate checklist

Before asserting pass:

1. Ran `evaluate_model` (or matrix wrap) with **`--ship-gates`**
2. Checked **all** policy suites (smoke, held_out, adversarial, ood, rico_held)
3. Used **placeholder_fidelity** (not soft `placeholder_validity`) for ship bars
4. Confirmed **meaningful parse** (not empty / useless stacks)
5. Confirmed **honest slot contract** — inventory from prompt/DESIGN.md, not
   hidden `gold.placeholders` when `honest_slot_contract=True`
6. Recorded **suite sizes** (especially `rico_held` n)
7. **Updated docs** via `documenting-experiment-results`

## Default honest bars (CLI `--ship-gates`)

| Suite | parse | structural | placeholder_fidelity | reward |
| --- | ---: | ---: | ---: | ---: |
| smoke | ≥ 0.66 | ≥ 0.35 | ≥ 0.25 | ≥ 0.30 |
| held_out | ≥ 0.40 | ≥ 0.30 | ≥ 0.15 | — |
| adversarial | ≥ 0.25 | ≥ 0.25 | — | — |
| ood | ≥ 0.25 | ≥ 0.25 | — | — |
| rico_held | ≥ 0.10 | ≥ 0.20 | — | — |

Do not lower these to green a run. Document a fail and change levers instead.

## Preferred honest recipes

```bash
# Focused honest fixture path (V6 champion stack)
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context

# Explicit ship-gates scoreboard
python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v1 \
  --model twotower --run-id <id> --ship-gates
```

Prefer `--matrix v4+` / V6+ rows (E35/E53 family) for honest claims. Historical
V3 template-fill clears that read gold placeholders are invalidated.

## Red flags

- Soft smoke-only gates treated as ship
- `rico_held` n=23 (or other stub) presented as the 1500 claim
- Reward inflated by gold DESIGN.md lint
- Curriculum / train seeds isomorphic to smoke (`smoke_align` era)
- "Pass" in old `gates.json` without re-eval after remediation

## After eval

**REQUIRED:** Follow `documenting-experiment-results` before claiming done.

## Checkpoints (full HF trains)

Production HF-context trains must land weights in
`hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/`
([checkpoint-bucket.md](../../../docs/design/checkpoint-bucket.md)). Confirm
`train_summary.json` → `checkpoint_bucket` (or an explicit
`--no-sync-checkpoints` / scratch rationale). Auth: `HF_TOKEN` /
`hf auth login`.

Then update [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) **and** the
README “Model card (summary)” with run id, URI, suite metrics, and claim level
(demo / scratch matrix / production HF). Ship claims without a card+summary
update are incomplete.
