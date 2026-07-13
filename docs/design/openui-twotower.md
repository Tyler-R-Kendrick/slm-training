# OpenUI TwoTower — Design Spec

## Problem

Build a small, on-device-friendly specialist that generates **placeholder-augmented OpenUI layout skeletons** from natural-language prompts. Literal copy is deferred to a separate copy model. The near-term goal is a **TwoTower** system (frozen AR context + trainable discrete diffusion denoiser) with grammar-constrained decoding — but **this cycle ships harnesses only**, not the model.

## Non-goals (this cycle)

- Implementing context tower, denoiser, cross-attention, or diffusion training
- Cactus runtime / custom kernels
- Full `thesysdev/openui` grammar and React runtime
- Awwwards scraping, RL/DPO, consistency distillation
- Training a production copy SLM

## Minimal OpenUI subset (v1)

Line-oriented assignments. Components: `Stack`, `Card`, `Text`, `Button`. Content uses scoped placeholders only (no free-form literals).

### BNF (informal)

```
program     ::= statement+
statement   ::= ident "=" expr
expr        ::= component | placeholder | ident
component   ::= ("Stack" | "Card" | "Text" | "Button") "(" arg_list? ")"
arg_list    ::= arg ("," arg)*
arg         ::= ident "=" value
value       ::= expr | string | number | bool
placeholder ::= ":" ident ("." ident)*
ident       ::= [A-Za-z_][A-Za-z0-9_]*
string      ::= '"' [^"]* '"'   # allowed only for non-content attrs (e.g. direction)
```

### Placeholder rules

- Content-bearing props (`text`, `label`, `title`, `body`) **must** be placeholders (`:hero.title`).
- Placeholder names are dotted scopes: `:section.body.p1`.
- Structural attrs (`direction`, `gap`, `variant`) may be literals.
- Parser rejects programs that put string literals in content props.

### Example

```
root = Stack(direction="vertical", children=hero)
hero = Card(title=:hero.title, body=:hero.body)
cta = Button(label=:cta.label)
```

## Future model architecture (spec only)

```
prompt → Frozen Context Tower (AR) → hidden states
                                      ↓ cross-attn
         Trainable Denoiser (masked/block diffusion) → OpenUI tokens
                                      ↓
                              CFG / DFA projection
                                      ↓
                         placeholder OpenUI program
```

- Context tower: small pretrained HF model, frozen (e.g. SmolLM2-135M).
- Denoiser: compact bidirectional Transformer over OpenUI token ids.
- Copy model: separate specialist filling placeholders (later).

## Three harnesses

Shared foundation: DSL parser/validator, placeholders, record schema.

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
| **Pipeline** | Load → map/placeholders → optional prompt synth → validate → dedupe → write |
| **Outputs** | `outputs/train_data/<version>/{manifest.json,records.jsonl,stats.json}` |
| **CLI** | `python -m scripts.build_train_data` |
| **Success** | All records parse; stats report counts; offline CI from fixtures |

### 2. Testing-data harness

| | |
| --- | --- |
| **Inputs** | Seed fixtures for eval suites; optional train manifest for leakage checks |
| **Suites** | `smoke`, `held_out`, `adversarial`, optional `ood` |
| **Outputs** | `outputs/test_data/<version>/{manifest.json,suites/<suite>/records.jsonl,stats.json}` |
| **CLI** | `python -m scripts.build_test_data` |
| **Success** | Suites frozen; no id overlap with provided train manifest |

### 3. Model-building harness

| | |
| --- | --- |
| **Inputs** | Train + test artifact paths from the data harnesses |
| **Role** | Config, loaders, `ModelPlugin` protocol, stub model, train/eval loops, metrics |
| **Outputs** | `outputs/runs/<run_id>/{checkpoints/,metrics.jsonl,eval.json}` |
| **CLIs** | `python -m scripts.train_model`, `python -m scripts.evaluate_model` |
| **Success** | Stub trains 1–2 steps and evaluates with `parse_rate` (no HF download) |

## Roadmap

1. Shared DSL + fixtures
2. Training-data harness
3. Testing-data harness
4. Model-building harness (stub)
5. **Later:** real TwoTower plug-in, richer data sources, grammar baking, Cactus export
