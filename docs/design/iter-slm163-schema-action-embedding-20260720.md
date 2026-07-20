# SLM-163 (SDE1-01): Schema-description action-embedding fixture (slm163-schema-action-embedding-20260720)

Matrix set: `slm163_schema_action_embedding`

Version: `sde1-01-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

Schema-derived action descriptions produce action embeddings that are more structured than random or stub initializations, as measured by coverage, nearest-neighbor cosine separation, sibling-family separation, and rare-vs-common centroid distance.

## Falsifier

Schema descriptions do not improve any of the above metrics over the current_stub baseline, or the shuffled control arm performs as well as the schema-driven arms.

## Arms

| Arm | Source | Promotable | Description |
| --- | --- | --- | --- |
| A_none | none | False | No action descriptions; embeddings remain randomly initialized. |
| B_current_stub | current_stub | False | Short stub glosses matching the existing teacher_init_embeddings path. |
| C_schema_description | schema_description | False | Schema-derived descriptions with property signatures and roles. |
| D_expanded_description | expanded_description | False | Rich teacher-style descriptions from the committed expanded JSON file. |
| E_shuffled | shuffled | False | Control arm: schema descriptions randomly reassigned to action keys. |

## Results

| Arm | Seed | d_model | Actions | Coverage | Mean NN cos | Sibling sep | Rare-common dist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_none | 0 | 32 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| B_current_stub | 0 | 32 | 79 | 1.000 | 0.402 | 1.059 | 0.804 |
| C_schema_description | 0 | 32 | 79 | 1.000 | 0.416 | 1.132 | 0.574 |
| D_expanded_description | 0 | 32 | 79 | 1.000 | 0.419 | 0.925 | 0.596 |
| E_shuffled | 0 | 32 | 79 | 1.000 | 0.416 | 1.015 | 0.660 |

## Go / no-go decision

**No-go for promotion.** Every arm is explicitly non-promotable. The harness proves the wiring and metrics plumbing over deterministic schema-derived descriptions, but it does not train or evaluate a real model. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained scorer and AgentV evaluation are available.

## Honest caveats

- The encoder is a deterministic hash projection, not a trained language model.
- Embeddings are perturbed by a tiny seed-dependent noise vector so that   different seeds are not degenerate.
- Sibling and rare/common metrics use hand-picked component families as a   sanity check, not a learned taxonomy.
- No Pareto or ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm163_schema_action_embedding_fixture --mode plan-only
python -m scripts.run_slm163_schema_action_embedding_fixture --mode fixture
```
