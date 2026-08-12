# SLM-577 — canonicalize metric_gaming archetypes + emptiness_probe tracing

**Linear:** [SLM-577](https://linear.app/quickdeploy-ai/issue/SLM-577/gh-1263-follow-up-canonicalize-metric-gaming-archetypes-emptiness)  
**Predecessor debt doc:** [`gh-1263-cd-remaining-debt-20260812.md`](gh-1263-cd-remaining-debt-20260812.md)  
**Honesty:** fixture/harness debt cleanup — not a ship claim.

## What changed

1. **`metric_gaming._archetypes()`** — adversarial positives and their derived
   negative transforms now use opaque `:slot_N` markers end-to-end. The shared
   owner asserts `assert_canonical_template_marker_inventory` on every returned
   `slot_contract`. Matching transforms in `oracle_scoring_replay.py` were
   rewritten to the same inventory so pred/gold/request cannot disagree on
   marker identity.
2. **`emptiness_probe.minimal_valid_program`** — rejected minimal candidates are
   logged at DEBUG (with exception type/message); if none validate, a WARNING
   lists every failure. Still returns `None` and skips honestly — no gate
   weakening, I6 intact.

## Consumers audited

| Surface | Action |
| --- | --- |
| `metric_gaming` case builders | Consume canonical archetypes; `MetricGamingCase.__post_init__` projection stays as defense in depth |
| `oracle_scoring_replay` transform tables + fixtures | Named markers projected to `:slot_N`; fixture builder keeps `canonicalize_example_template_markers` |
| `slm186_verified_utility` | Uses `build_all_cases` / `evaluate_metric_gaming` — inherits canonical markers |
| `slm172_render_equivalence` | `metric_gaming_minimal_valid` arm pulls trap cases; independent `:card.*` base program left alone (render-equivalence fixture, not GenerationRequest inventory) |

## Verification

```bash
PYTHONPATH=src uv run pytest \
  tests/test_evals/test_metric_gaming.py \
  tests/test_evals/test_oracle_scoring_replay.py \
  tests/test_evals/test_emptiness_probe.py -q
PYTHONPATH=src uv run python -m scripts.verify_merge_ready --fast
```

Measured locally on this branch: metric_gaming + oracle + emptiness probe tests
green after the rewrite.
