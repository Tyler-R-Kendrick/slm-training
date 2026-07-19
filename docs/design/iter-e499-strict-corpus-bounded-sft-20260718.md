# E499 — bounded strict-corpus SFT

E499 tests whether the repaired strict fixture corpus improves a tiny,
matched frozen-HF/choice-codec training run. It does not. The strict-r4
checkpoint and a choice-compatible integrated follow-up both regress smoke
structure and component recall versus the diverse-root control.

## Reproducibility

- Hard policy: every process was capped at 170 seconds; train summaries record
  `max_wall_minutes=3.0`.
- Device/context: CPU, frozen local
  `HuggingFaceTB/SmolLM2-135M`, 134,515,008 frozen parameters.
- Trainable model: TwoTower, 230,210 parameters, `d_model=64`, two heads,
  one context layer, two denoiser layers, choice output codec.
- Matched budget: seed 0, batch 4, learning rate `3e-4`, 1,000 target tokens,
  no DESIGN context, no full-state checkpoint.
- Evaluation: honest constrained `smoke`, `n=1`, four generation steps,
  96-token LTR cap, one attempt, no unconstrained fallback.
- Persistence: every successful train wrote `last.pt`, tokenizer sidecars,
  `train_summary.json`, telemetry, and a trace. Every final matched evaluation
  emitted AgentEvals JSONL and a pinned AgentV bundle.
- Storage: explicit `--no-sync-checkpoints`; these are local negative scratch
  diagnostics, not full HF-context bucket candidates.

## Corpus diagnosis

The strict-r4 corpus has 104 records but only 20 root parents, with six
records per parent at p95. The 108-record control spans 94 root parents and
includes 61 language-contract records. That concentration motivated a strict
integrated r5 build.

r5 admitted 235 records across 65 program and 132 structural families with no
decontamination hits, but 76 targets were fragments that the document-only
choice codec cannot encode. This is a real task/codec compatibility boundary,
not a quality-gate failure. A follow-up r6 build disabled only fragment-emitting
language-contract and scope-corpus producers. It passed a direct 67/67 choice
codec preflight, but synthesis feedback still reported high rejection and
redundant paraphrase expansion, so r6 remained unpublished.

## Matched result

| Arm | Records | Steps / target tokens | Last loss | Syntax | Meaningful | Fidelity | Structure | Component recall | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Diverse-root control r4 | 108 | 10 / 1,023 | 30.8688 | 1.0 | 0.0 | 0.0 | 0.1542 | 0.25 | 6,853 ms |
| Strict fixture r4 | 104 | 9 / 1,034 | 32.2048 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 7,020 ms |
| Choice-compatible strict r6 | 67 | 9 / 1,091 | 32.8781 | 1.0 | 0.0 | 0.0 | 0.0375 | 0.0 | 5,881 ms |

All three evaluations fail `no_placeholders`; reward is zero and AgentV is
0/1 for each arm. Relative to control, both candidates lose `0.1167` structure
and `0.25` component recall. r6 is 972.65 ms faster, but quality gates outrank
that latency gain.

## Invalid exploratory arms

Four earlier checkpoints are retained and registered as invalid evidence:
lexer r1 changed parameter count through data-derived vocabularies; choice r2
had no valid control because its control corpus contained 61 fragment targets;
scratch-choice r3 still changed parameter count through its context vocabulary.
Only the frozen-HF r4 pair is a matched corpus comparison.

## Decision

Keep the strict fixture repairs and the surfaced fragment/choice-codec
compatibility diagnosis. Do not replace `remediated_roots`, publish r6, sync an
E499 checkpoint, promote a model, or make a ship claim. The next data iteration
must preserve broad independent roots while reducing redundant paraphrases and
must declare codec-compatible target kinds before training.

Canonical metrics and every checkpoint SHA:
[JSON](iter-e499-strict-corpus-bounded-sft-20260718.json).
