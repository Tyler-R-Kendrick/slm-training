# Online diffusion data adapter

## Status and claim boundary

Implemented by SLM-14 as an opt-in TwoTower training path. This is data and
model plumbing, verified by unit tests; it is **not** a trained checkpoint or a
ship-readiness result. Any quality claim still requires the full scoreboards and
`--ship-gates` described in the quality matrix.

## Contract

`src/slm_training/data/diffusion/` converts each clean token sequence into a
fresh in-memory corruption at batch time. Clean examples and the target-token
cache remain unchanged; the adapter never writes or materializes masked corpus
rows.

Each `DiffusionCorruption` carries aligned target/noisy canvases, a loss mask,
insert/delete state, source and target lengths, a target-length bucket, and
canvas-aligned auxiliary labels. Calling `reconstruct()` with the target values
must recover the clean target after learned deletion slots are removed.

## Policy mixture

The online mixture contains all required policy families:

| Policy | Training state |
| --- | --- |
| `uniform` | Random token subset |
| `contiguous` | One contiguous token span |
| `statement` | Whole newline-delimited statement |
| `ast_subtree` | Balanced delimiter subtree |
| `reference` | One binder definition and its uses |
| `edit_local` | ProgramSpec `meta.edit.changed_token_indices` or `changed_span` |
| `disjoint` | At least two separated mask islands when the row permits |
| `all_mask` | Full non-padding canvas |
| `expansion` | Insertion positions represented by `<mask>` source states |
| `contraction` | Extra source positions learn a target `<pad>` deletion state |
| `reorder` | Statement or span order corruption |

`align_token_edits(source_ids, target_ids, ...)` uses sequence alignment for
real long-to-short and short-to-long edit pairs. It overallocates only as much
canvas as the aligned pair needs, represents statement insertion as insertion
states, and fails explicitly when the aligned canvas exceeds `max_length`.

## Variable-length prediction

With `mask_pattern="diffusion"`, TwoTower adds a context-pooled target-length
classification head. Training adds bucket cross-entropy with
`diffusion_length_loss_weight`; generation uses the predicted bucket upper bound
as each row's MaskGIT canvas unless the caller supplies an explicit `max_len`.
Expansion and contraction loss remain token-aligned because noisy and target
canvases have the same shape, including target-padding deletion positions.

Default bucket upper bounds are `32,64,96,128,192,256,384,512` and are clamped
to `max_target_len` at inference.

## Optional aligned supervision

The adapter currently emits token-aligned labels for statement boundaries, AST
node category, grammar-production category, identifier definition/use role,
component type, argument position, expected target length, changed/preserved
state, and (when supplied by record metadata) verifier error category. These
labels are carried by the adapter for later auxiliary heads; SLM-14 does not add
losses for them.

## Configuration

Programmatic configuration is available on `ModelBuildConfig` and
`TwoTowerConfig`. The training CLI exposes the same opt-in path:

```bash
python -m scripts.train_model \
  --mask-pattern diffusion \
  --output-tokenizer lexer \
  --diffusion-policies uniform,contiguous,statement,ast_subtree,reference,edit_local,disjoint,all_mask,expansion,contraction,reorder \
  --diffusion-length-buckets 32,64,96,128,192,256 \
  --diffusion-overallocate 8 \
  --diffusion-length-loss-weight 0.1 \
  <normal train and checkpoint-sync arguments>
```

This command is illustrative. A real run triggers the repository's experiment
documentation, checkpoint-bucket, and model-card requirements.

## Verification

`tests/test_data/test_diffusion.py` covers every policy against both the
compositional and lexer-native tokenizers, explicit reference groups and
disjoint holes, real expansion/contraction reconstruction, aligned auxiliary
labels, online resampling without clean-row mutation, and length-head gradient
flow/cache preservation.
