# G4 sketch-of-thought reasoning harness (SLM-36)

Status: implemented fixture-grade harness — **not a ship claim**. Evidence:
[iter-g4-reasoning-bench-20260717.md](iter-g4-reasoning-bench-20260717.md).

## What this is

A checkable-answer reasoning bench where the trace is a program in a
task-specific DSL and the scorer is a deterministic evaluator — the first
checkable-answer machinery in this repo (the existing `task_scoreboard` only
does OpenUI structure and normalized text equality).

- **DSL**: `arith-sketch` (`dsl/grammars/arith_sketch.lark`) — straight-line
  arithmetic bindings; the final `root` binding is the answer. Backend via
  the generic `LarkFileBackend` (zero new parser code); registered as the
  fifth grammar backend and the third F1 pack
  (`dsl/packs/arith_sketch.py`).
- **Deterministic expansion for bound spans**: the generic transformer's
  resolve pass substitutes refs into the root expression; the pack's
  evaluator computes the number. Validity and answer scoring are one code
  path (`evaluate_answer`) — no separate judge to drift. Invalid traces
  score wrong, fail-closed, no repair.
- **Bench** (`harnesses/reasoning/bench.py`, CLI
  `scripts/run_reasoning_bench.py`): two matched arms, same tiny scratch
  TwoTower (compositional corpus-derived tokenizer — DSL-agnostic), same
  word-problem corpus from the pack generator, same numeric scorer:
  - *sketch arm* — PAL/PoT-**Adapted** (reason-in-formal-language +
    deterministic execution, on a **trained tiny model** instead of a
    prompted frozen LLM): targets are programs; emitted trace is executed.
  - *direct arm* — no-trace control: targets are the bare numeric answer.

## Positioning (per the source registry)

Sketch-of-Thought (Aytes et al., arXiv:2503.05179) is prompt-level
NL-symbolic sketching with a frozen large model; PAL (arXiv:2211.10435) and
PoT (arXiv:2211.12588) are Python-emitting frozen LLMs. The
trained-tiny-model + externalized-grammar + deterministic-expansion variant
is the unclaimed combination this harness instantiates. A frozen-large-LLM
PAL baseline is out of honest tiny-scale scope — the in-scope baseline is
the matched direct-answer arm.

## Stated boundaries

- Decode is **unconstrained** parallel MaskGIT in both arms: the fastpath
  grammar gate and lexer tokenizer are OpenUI-hard, so wiring the
  incremental engine (`OpenUIIncrementalEngine(arith_sketch.lark)` — the
  engine itself already accepts the grammar) into non-OpenUI constrained
  decode is the tracked follow-up. The fixture failure modes (forward refs,
  missing `root`) are precisely what that integration would eliminate.
- The corpus is template-generated word problems (6 templates, 2–3 step
  programs); real reasoning benchmarks (GSM-style) are follow-up scale.
