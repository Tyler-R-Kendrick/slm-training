# SPV2-01: Semantic-contrast corpus v1

A versioned, hard-valid semantic-contrast corpus for OpenUI.  Every record is
parser / schema / reference valid, but the negative side is semantically wrong
with respect to an explicit prompt-component contract.  The corpus is built on
top of `SemanticPlanV1` so that corruptions are plan-level and reproducible,
not regex hacks.

**Deliverables**

- Builder module: `src/slm_training/data/semantic_contrast/`
- CLI: `scripts/build_semantic_contrasts.py`
- Tests: `tests/test_data/test_semantic_contrast.py`
- Artifact: `outputs/data/eval/semantic_contrast_v1/`
- Scoreboard: `docs/design/semantic-contrast-corpus-v1.json`

## Corruption taxonomy

| Family | Transform | Severity | Semantics broken |
| --- | --- | --- | --- |
| `content` | `content_swap_family` | moderate | Change a content role's component family while keeping the binding. |
| `content` | `content_invert_role` | moderate | Flip a symbol's semantic role to an incompatible prop. |
| `topology` | `topology_delete_leaf` | severe | Remove a leaf role and its binding from the rendered program. |
| `topology` | `topology_reparent` | severe | Move a child to a different parent. |
| `binding` | `binding_swap_symbol` | severe | Rebind a role to a different existing symbol. |
| `binding` | `binding_introduce_incompatible_symbol` | severe | Inject a new placeholder with a mismatched semantic role. |
| `contract` | `contract_unresolve` | severe | Mark a requirement unresolved and remove the role that satisfied it. |
| `contract` | `contract_archetype_mismatch` | benign | Contradict the prompt archetype. |
| `positive` | `positive_control_identity` | benign | Original source compiled unchanged. |

## Build recipe

```bash
python -m scripts.build_semantic_contrasts --dataset-id semantic_contrast_v1
```

- Seed: `0`
- Source ProgramSpecs: `12` (filtered for at least one non-`Stack` component and
  at least one placeholder)
- Components sampled: `TextContent`, `Button`
- Max depth/width: `2` / `3`
- Splits: `train` (80 %) / `held_out` (20 %)
- Honesty mode: `production`
- Admission gate: verifier pass + positive `binding_aware_meaningful_v2` pass +
  negative `binding_aware_meaningful_v2` fail

## v1 scoreboard

See [`semantic-contrast-corpus-v1.json`](semantic-contrast-corpus-v1.json) for
the full stamped payload.  Headline numbers:

| Family | n_total | n_admitted | verifier pass | meaningful pass | false negative |
| --- | --- | --- | --- | --- | --- |
| binding | 14 | 14 | 1.00 | 0.00 | 0.00 |
| content | 24 | 24 | 1.00 | 0.00 | 0.00 |
| contract | 14 | 14 | 1.00 | 0.00 | 0.00 |
| topology | 2 | 2 | 1.00 | 0.00 | 0.00 |
| positive | 78 | 66 | 1.00 | 1.00 | 0.00 |

*Positive `n_total` = 66 positive-control records + 12 negative-side control
records; `n_admitted` = 66 positive controls that pass meaningful eval.*

## Honest caveats

- This is a **fixture/data artifact**, not a production model claim.  It
  provides a reusable negative corpus and a baseline scoreboard; it does not
  clear `--ship-gates` for a trained model.
- The topology transform family is currently small (2 admitted) because many
  plan-level topology edits produce seeds that fail the OpenUI schema's
  required-children constraints.  Future work can add container-aware topology
  transforms once `PlanSeedBuilder` supports direction defaults or once we add
  a post-compile repair step.
- The corpus is scoped to scalar-content components (`TextContent`, `Button`)
  to keep the plan compiler valid.  Extending to containers and forms is a
  follow-up data-builder task.
- The `contract_archetype_mismatch` transform is benign because the current
  prompt contract does not yet include an explicit archetype assertion; it is
  kept as a taxonomy placeholder.

## Version stamp

- Component `data.semantic_contrast`: `v1`
- Component `evals.meaningful_program`: `2.0.0`
- Code commit: `9792cab2a43e9dfd6b15ece005a0215b5c3a480e` (dirty worktree)
