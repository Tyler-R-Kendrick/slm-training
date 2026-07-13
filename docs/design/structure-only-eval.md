# Structure-only scaffolds (no style eval)

OpenUI **scaffold gold** is layout structure only:

- components + nesting + direction (`column` / `row`)
- placeholder slots (`:ns.slot`)

Not scaffold gold (stripped from fixtures / ignored in eval):

- gap tokens (`s`, `m`, `l`, `2xl`, …)
- typography sizes (`large-heavy`, …)
- color-role variants (`primary`, `secondary`, …)
- DESIGN.md colors / type scale (context only)

## Scrubbing

- `slm_training.data.structure.strip_style_literals`
- Applied in train/test `_normalize` pipelines and RICO / Awwwards generators
- `normalize_openui_structure` also strips style for leakage fingerprints

## Eval

| Metric | Style? |
|--------|--------|
| `structural_similarity` | No — style args stripped first |
| `placeholder_fidelity` | Binding only |
| `reward_score` | Structure-only composite (`design_md=None`) |
| `gold_design_lint_score` | Diagnostic on gold DESIGN.md — **not** ship |

`--fail-under-design-lint` is ignored when `--ship-gates` is set so unused-color
warnings cannot fail readiness. Quality filters only soft-penalize DESIGN.md
**errors**, not warnings.

## Fixtures

`fixtures/test_seeds.jsonl` and `fixtures/train_seeds.jsonl` are structure-scrubbed.
Rebuild corpora after pulling:

```bash
python -m scripts.build_train_data ...
python -m scripts.build_test_data ...
```
