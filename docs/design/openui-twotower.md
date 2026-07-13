# OpenUI TwoTower — Design Spec

## Problem

Build a small, on-device-friendly specialist that generates **placeholder-augmented OpenUI layout skeletons** from natural-language prompts. Literal copy is deferred to a separate copy model. The near-term goal is a **TwoTower** system (frozen AR context + trainable discrete diffusion denoiser) with grammar-constrained decoding — but **harnesses ship first**, not the model.

## Non-goals (current cycle)

- Cactus runtime / custom kernels
- Full React rendering stack in Python (use `@openuidev/react-lang` later for demos)
- Awwwards scraping, RL/DPO, consistency distillation
- Training a production copy SLM

## Official OpenUI Lang (source of truth)

Parsing, serialization, and system-prompt generation use **`@openuidev/lang-core`** via the Node bridge in [`tools/openui_bridge/`](../../tools/openui_bridge/).

| Capability | Official API |
| --- | --- |
| Parse | `createParser(library.toJSONSchema()).parse(source)` |
| Serialize | `jsonToOpenUI(root, library)` |
| System prompt | `library.prompt({...})` / `generatePrompt` |
| Library | `defineComponent` + `createLibrary` (Zod props) |

Python harnesses call [`src/slm_training/dsl/lang_core.py`](../../src/slm_training/dsl/lang_core.py), which shells to `tools/openui_bridge/cli.mjs`.

### Training subset library

Defined in [`tools/openui_bridge/library.mjs`](../../tools/openui_bridge/library.mjs):

- `Stack(children, direction?, gap?)`
- `Card(title, body?)`
- `Text(content)`
- `Button(label)`

Root component: `Stack`.

### Placeholder policy (ours, on top of lang-core)

Official OpenUI allows arbitrary strings. For this project, content props (`title`, `body`, `content`, `label`) **must** be placeholder strings:

```
root = Stack([hero], "vertical")
hero = Card(":hero.title", ":hero.body")
```

Enforced in the bridge after parse (`policy_errors`). Free-form copy is rejected so a separate copy model can fill placeholders later.

### Example (canonical OpenUI Lang)

```
root = Stack([hero, cta], "vertical")
hero = Card(":hero.title", ":hero.body")
cta = Button(":cta.label")
```

Export the official teacher prompt with:

```bash
python -m scripts.export_openui_prompt
```

## Model architecture (implemented)

```
prompt → Context Tower (scratch TokenEncoder | frozen HF e.g. SmolLM2) → hidden states
                                      ↓ cross-attn
         Trainable Denoiser (MaskGIT-style masked diffusion) → OpenUI tokens
                                      ↓
         streaming parser grammar guard (createStreamingParser) + placeholder policy
                                      ↓
                         placeholder OpenUI program
```

- **Context tower**: `scratch` (default) or `hf` via `--context-backend hf --hf-model HuggingFaceTB/SmolLM2-135M` (frozen by default; only the projection trains).
- **Denoiser**: bidirectional Transformer with cross-attention (`DenoiserTower`), trained with random span masking.
- **Tokenizer**: `OpenUITokenizer` for OpenUI targets; HF models use their own prompt tokenizer.
- **Grammar-constrained decode**: structural logit bias + remask on hard streaming errors + LTR repair filtered by official `createStreamingParser`.
- **Copy model**: separate specialist filling placeholders (later).

Package: [`src/slm_training/models/`](../../src/slm_training/models/).

## Three harnesses

Shared foundation: official lang-core bridge + placeholders + `ExampleRecord` schema.

### Record schema

```json
{
  "id": "string",
  "prompt": "string",
  "openui": "string",
  "placeholders": ["string"],
  "split": "train|held_out|smoke|adversarial|ood",
  "source": "string",
  "meta": {}
}
```

### 1. Training-data harness

| | |
| --- | --- |
| **Inputs** | **RICO** semantic screens (default) and/or seed fixtures |
| **Pipeline** | Load RICO → map to placeholder OpenUI → optional prompt synth → **lang-core validate** → canonicalize → dedupe → write fingerprints |
| **Outputs** | `outputs/train_data/<version>/{manifest.json,records.jsonl,stats.json}` |
| **CLI** | `python -m scripts.build_train_data --source rico` |

RICO source: Hugging Face [`shunk031/Rico`](https://huggingface.co/datasets/shunk031/Rico) semantic annotations, converted into the OpenUI subset (`Stack` / `Card` / `Text` / `Button`) with placeholders. Local slices live under [`fixtures/rico/`](../../fixtures/rico/).

### 2. Testing-data harness

| | |
| --- | --- |
| **Inputs** | RICO **test** split fixtures + hand-written adversarial/ood; **required** train manifest |
| **Suites** | `smoke`, `held_out`, `adversarial`, `ood` |
| **Leakage** | Rejects any test record whose **id**, **prompt**, **openui**, or pair fingerprint appears in train |
| **Outputs** | `outputs/test_data/<version>/suites/<suite>/records.jsonl` |
| **CLI** | `python -m scripts.build_test_data --train-manifest ...` |

### 3. Model-building harness

| | |
| --- | --- |
| **Inputs** | Train + test artifact paths |
| **Role** | Config, loaders, `ModelPlugin`, **TwoTower** (default) + stub, train/eval loops |
| **Eval** | `parse_rate` via lang-core validate; placeholder fidelity; canonical serialize match |
| **CLIs** | `python -m scripts.train_model`, `python -m scripts.evaluate_model` |

## Roadmap

1. Official lang-core bridge + fixtures (done)
2. Training / testing / model-build harnesses (done)
3. GPU multi-farm MCP for cheap training pods (done)
4. TwoTower plug-in — context + masked denoiser (done)
5. RICO train data + strict leakage checks (done)
6. **HF frozen context tower + streaming/grammar-constrained decode (this revision)**
7. **Later:** richer `@openuidev` library, React demo via `@openuidev/react-lang`, larger RICO slices / GPU farms
