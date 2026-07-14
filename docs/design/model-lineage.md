# Canonical two-track model lineage

This is the operational contract for production model iteration. E/X/P matrix
rows remain ablation evidence; they are not deployable identities.

## Equal production tracks

| Track | Frozen base | Branch artifact | Deployment artifact |
| --- | --- | --- | --- |
| `twotower` | E53 recipe + `HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | complete child weights | quantized TwoTower ONNX, ≤1GB |
| `causal_lm` | winner of the pinned Qwen bakeoff | LoRA adapter | merged-LoRA ONNX/GGUF, ≤1GB |

The causal bakeoff candidates are permanently pinned to
`Qwen/Qwen2.5-Coder-0.5B-Instruct@ea3f2471cf1b1f0db85067f1ef93848e38e88c25`
and `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
`model-cycle lock-causal-base` selects by full gate pass, semantic plus
structural score, warm latency, then artifact size. Once written, the base
pointer cannot move to another model or revision.

The TwoTower recipe is the full E53 stack: width 192, six heads, three context
layers, six denoiser layers, frozen HF context, lexer tokenizer, symbol table,
factorized embeddings, mixed statement masking, schema and honest slot
contract, 16 denoising steps, and combined CoRe/gate/entropy remasking. The
causal recipe is LoRA rank 16, alpha 32, dropout 0.05, all attention/MLP
projections, length 512, effective batch 32, LR 2e-4, cosine decay, 3% warmup,
and early stopping ([LoRA](https://arxiv.org/abs/2106.09685)).

## Records and storage

`src/slm_training/lineage/` owns frozen `RunManifest`, `DataSnapshot`,
`EvaluationReport`, `MergeManifest`, and `ChampionPointer` records. Canonical
JSON is SHA-256 content-addressed under `outputs/lineage/`:

```text
outputs/lineage/
  runs/<run_id>/{manifest.json,revisions/,current.json}
  data_snapshots/<snapshot_id>-<sha>.json
  evaluations/<report_id>-<sha>.json
  merges/<merge_id>-<sha>.json
  tracks/<track>/base/{history/,current.json}
  champions/<track>/{history/,current.json}
  deployments/<track>/{history/,current.json}
  deployments/selected.json
```

Records and artifacts are create-only. Only `current.json` and
`deployments/selected.json` are replaced, using same-directory atomic rename.
Lifecycle transitions are append-only revisions: `running → screened →
validated → champion → deployed`, with terminal `rejected` available before
champion.

`resume` accepts only the same run's native full-state checkpoint:
`last_full_state.pt` for TwoTower or a Transformers `checkpoint-*` directory
for causal LoRA. Both restore model, optimizer, RNG, and sampler. `branch`
reads a validated parent's model weights, creates a fresh run directory, and
starts a new optimizer. It never consumes the parent's optimizer state and it
rejects architecture, tokenizer, or adapter-shape changes.

## Corpus and evaluation snapshots

`snapshot-data` hashes curated/RICO/Awwwards sources and annotation inputs.
Feedback conversion preserves generation, reviewer, annotator, correction,
provider/model, prompt, parent, and generation identities. Approved or
corrected valid output becomes SFT data. Same-prompt valid up/down output can
become DPO data. Invalid output is verifier-negative only. Incremental SFT uses
10% validated champion replay ([On-Policy Replay](https://arxiv.org/abs/2605.29495)).
The trigger helpers hold SFT until 25 new positives and DPO until 100 valid
pairs.

`snapshot-eval` fails closed unless smoke, held-out, adversarial, OOD, all
1,500 `rico_held` records, and a nonempty never-trained human-feedback holdout
are present. The 19-example remediated corpus remains screening-only.

## Cycle CLI

```bash
model-cycle snapshot-data --snapshot-id train-v1 --source outputs/train_data/v1
model-cycle snapshot-eval --snapshot-id eval-v1 \
  --suite smoke=outputs/test_data/v1/smoke.jsonl \
  --suite held_out=outputs/test_data/v1/held_out.jsonl \
  --suite adversarial=outputs/test_data/v1/adversarial.jsonl \
  --suite ood=outputs/test_data/v1/ood.jsonl \
  --suite rico_held=outputs/test_data/v1/rico_held.jsonl \
  --human-feedback-holdout fixtures/annotations/human_holdout.jsonl

model-cycle init --track twotower --run-id tt-baseline \
  --data-snapshot-sha <sha> --eval-snapshot-sha <sha>
model-cycle branch --parent tt-baseline --run-id tt-cycle-001
model-cycle train --run-id tt-cycle-001 --train-dir outputs/train_data/v1 \
  --target-token-count <snapshot-target-tokens> --token-rung 0.5
model-cycle evaluate --run-id tt-cycle-001 --weighted-nll <value> \
  --warm-p95-seconds <windows-p95>
model-cycle export --run-id tt-cycle-001 \
  --output outputs/exports/tt-cycle-001 --format onnx
model-cycle promote --run-id tt-cycle-001 --report <sha> \
  --parent-report <sha> --finalist-report <sha> --finalist-report <sha>
model-cycle deploy --track twotower
```

Each cycle is one control plus at most two single-lever candidates. Screen one
seed at 0.5x, 1x, and 3x target-token rungs; finalists use three seeds. Local
training uses DirectML where the lower trainer supports it. NPU/WebNN stays
inference-only. Promotion jobs may use HF Jobs with Trackio but remain
incomplete until their checkpoint URI is persisted to the OpenUI bucket.

## Promotion, merge, and deployment

Promotion requires every honest ship gate, lower weighted NLL than the parent,
≤2% binding/structural/repair NLL regression, ≤2-point parse/fidelity/request
coverage/structural regression, stable ranking over three seeds and the two
largest rungs, ≤1GB quantized output, and Windows warm 256-token p95 ≤15s.

The web lineage layer exposes exact deployment manifests and blinded pair
creation/voting; API/UI adapters can use those stores without learning the
hidden side-to-model mapping.
Deployment additionally requires at least 100 comparisons, candidate win rate
above 55%, and a 95% Wilson lower bound above 50%.

Merge accepts only siblings with identical track, base revision, architecture,
tokenizer, common parent, and parameter shapes. It evaluates arithmetic
averaging ([Model Soups](https://arxiv.org/abs/2203.05482)) and TIES
([TIES-Merging](https://arxiv.org/abs/2306.01708)). A merged artifact is a new
screened challenger; it never updates a champion automatically and tracks are
never merged together.

Both plugins use the official OpenUI validator. TwoTower uses the existing DFA
fast path; causal generation uses persistent prefix token-mask caching and one
grammar-constrained generation pass, not retry repair (the caching design is
modeled on [XGrammar](https://arxiv.org/abs/2411.15100)).

Historical E/X/P artifacts import as `legacy_evidence` with unknown parents.
The five-step DirectML artifact imports as `hardware_smoke`. Neither kind can
enter a champion pointer. No usable historical production checkpoint exists,
so the first champions must be newly trained and fully evaluated.
