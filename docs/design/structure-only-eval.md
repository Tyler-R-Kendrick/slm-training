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
| `placeholder_fidelity_normalized` | Binding with namespace segment stripped (ablation) |
| `reward_score` | Structure-only composite (`design_md=None`) |
| `gold_design_lint_score` | Diagnostic on gold DESIGN.md — **not** ship |

`--fail-under-design-lint` is ignored when `--ship-gates` is set so unused-color
warnings cannot fail readiness. Quality filters only soft-penalize DESIGN.md
**errors**, not warnings.

## Slot contract conditioning (F2)

Eval records expose a **placeholder inventory** (`record.placeholders`) — spec-level
content slots, not layout leakage. When `slot_contract_in_context` is enabled, the
inventory is appended to context as `---SLOT_CONTRACT---` so the model knows which
`:namespace.slot` bindings to emit. Optional `slot_contract_constrained_decode`
restricts placeholder token emission to the contract during grammar decode.

Ship gates still use strict `placeholder_fidelity` (exact overlap). The contract
makes that objective well-posed: without it, eval namespaces (`:smoke.*`) are
unknowable from the prompt alone even with compositional tokenization.

## Train/test structural disjointness

`build_test_data` rejects fixture records whose **layout tree** (style-stripped,
placeholder-normalized) appears in the train manifest. Train synthesis can
accidentally recreate test fixture structures (e.g. dual-card column, tabs panel).

`build_train_data` loads reserved structure fingerprints from
`src/slm_training/resources/test_seeds.jsonl` and drops any train record (including synthesizer
variants) that collides. Stats report `structure_reserved_rejected`.

```bash
python -m scripts.build_train_data --source all --version v1
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json
```

## Tokenizer v2 checkpoint migration

Compositional placeholder tokenization (tokenizer v2) invalidates v1 checkpoints.
To salvage transformer weights (not placeholder embeddings), rebuild vocabulary
from train records and remap shared token rows:

```bash
python -m scripts.migrate_checkpoint \
  --checkpoint outputs/runs/legacy_run/checkpoints/last.pt \
  --train-records outputs/data/train/v1/records.jsonl \
  --output outputs/runs/legacy_run_v2/checkpoints/last.pt
```

Fresh training from scratch is still preferred for ship gates.

## Fixtures

`src/slm_training/resources/test_seeds.jsonl` and `src/slm_training/resources/train_seeds.jsonl` are structure-scrubbed.
Rebuild corpora after pulling:

```bash
python -m scripts.build_train_data ...
python -m scripts.build_test_data ...
```
