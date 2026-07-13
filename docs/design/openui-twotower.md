# OpenUI TwoTower — Design Spec

## Problem

Build a small, on-device-friendly specialist that generates **placeholder-augmented OpenUI layout skeletons** from natural-language prompts. Literal copy is deferred to a separate copy model. The near-term goal is a **TwoTower** system (frozen AR context + trainable discrete diffusion denoiser) with grammar-constrained decoding — but **harnesses ship first**, not the model.

## Non-goals (current cycle)

- Implementing context tower, denoiser, cross-attention, or diffusion training
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

## Future model architecture (spec only)

```
prompt → Frozen Context Tower (AR) → hidden states
                                      ↓ cross-attn
         Trainable Denoiser (masked/block diffusion) → OpenUI tokens
                                      ↓
                   official parser + placeholder policy / CFG
                                      ↓
                         placeholder OpenUI program
```

- Context tower: small pretrained HF model, frozen (e.g. SmolLM2-135M).
- Denoiser: compact bidirectional Transformer over OpenUI token ids.
- Copy model: separate specialist filling placeholders (later).
- Prefer official `createStreamingParser` when wiring diffusion unmask streams.

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
| **Inputs** | Seed fixtures; optional external path config |
| **Pipeline** | Load → optional prompt synth → **lang-core validate** → canonicalize via `jsonToOpenUI` → dedupe → write |
| **Outputs** | `outputs/train_data/<version>/{manifest.json,records.jsonl,stats.json}` |
| **CLI** | `python -m scripts.build_train_data` |

### 2. Testing-data harness

| | |
| --- | --- |
| **Inputs** | Eval fixtures; optional train manifest for leakage checks |
| **Suites** | `smoke`, `held_out`, `adversarial`, `ood` |
| **Outputs** | `outputs/test_data/<version>/suites/<suite>/records.jsonl` |
| **CLI** | `python -m scripts.build_test_data` |

### 3. Model-building harness

| | |
| --- | --- |
| **Inputs** | Train + test artifact paths |
| **Role** | Config, loaders, `ModelPlugin`, stub model, train/eval loops |
| **Eval** | `parse_rate` via lang-core validate; placeholder fidelity; canonical serialize match |
| **CLIs** | `python -m scripts.train_model`, `python -m scripts.evaluate_model` |

## Roadmap

1. Official lang-core bridge + fixtures (this revision)
2. Training / testing / model-build harnesses (done)
3. GPU multi-farm MCP for cheap training pods (done)
4. **Later:** real TwoTower plug-in, richer library (more `@openuidev` components), streaming parser for diffusion, React demo via `@openuidev/react-lang`
