<!-- rtk-instructions v2 -->
# RTK — Token-Optimized CLI

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
cargo test                 rtk cargo test
docker ps                  rtk docker ps
kubectl get pods           rtk kubectl pods
```

## Meta commands (use directly)

```bash
rtk gain              # Token savings dashboard
rtk gain --history    # Per-command savings history
rtk discover          # Find missed rtk opportunities
rtk proxy <cmd>       # Run raw (no filtering) but track usage
```
<!-- /rtk-instructions -->


<!-- token-efficiency-stack -->
# Token-efficiency stack (project)

Canonical skills: `.agents/skills/` (Claude / Codex / Cursor / GitHub Copilot).

| Tool | Role |
| --- | --- |
| **ponytail** | Minimal code — YAGNI ladder before writing |
| **caveman** | Opt-in terse chat (`/caveman`); code/commits stay normal |
| **headroom** | Compress large tool outputs in-context |
| **rtk** | Compress shell command output (`rtk <cmd>`) |

Always-on for Copilot Chat: prefer minimal diffs (ponytail ladder) and `rtk` for verbose CLI. Activate caveman/headroom skills when needed.
Before creating or relocating tracked paths, activate `organize-repository`,
follow `docs/repository-organization.md`, and use `git mv` for moves.
<!-- /token-efficiency-stack -->


<!-- decode-invariants -->
# Canonical instructions

Follow [`AGENTS.md`](../AGENTS.md) — the canonical instructions for every coding
agent in this repo. Prefer it over tool-specific defaults on process conflicts.

# Non-negotiable architecture invariants (goal law)

This repo is **not training a natural-language LLM.** It trains a
**grammar-constrained symbolic diffusion model** that outputs templated grammars
(scaffolded structure + structural reasoning). Templated / natural-language
content is deferred to a real external LLM (symbol-only output contract,
`dsl/language_contract.py`, `OUTPUT_CONTRACT_VERSION=2`). Everything below
constrains every model, harness, lever, experiment, doc, and agent action here.
Canonical expansion with file pointers and current status:
[`docs/design/decode-invariants.md`](../docs/design/decode-invariants.md).

1. **Never output invalid grammar.** Every production decode path is
   grammar-constrained end to end. Unconstrained arms
   (`--unconstrained-control`, HTTP `grammar_constrained=false`, eval `raw`
   arms) are diagnostic controls only — never production defaults, never
   serving paths, never shipped, certified, or gated on.
2. **Fail closed.** Production configs must not set
   `allow_unconstrained_fallback=True`, must run finalize validation, and must
   raise rather than return uncertified text — in every backend, ONNX included.
   An empty legal domain is a dead end, never a full-vocabulary fallback.
3. **No lever, experiment, or skill may weaken constrained decoding.** Levers
   may change *how legal symbols are chosen* — never *whether output is legal*.
   Weakening levers live in `levers.CONSTRAINT_WEAKENING_LEVERS` and are
   CI-blocked from production configs.
4. **Deterministic completion paths bypass inference** wherever a deterministic
   answer exists. Deterministic decode proofs outrank learned, semantic,
   confidence, and preference scores.
5. **Forced bypass on singletons:** exactly one valid next symbol means commit
   it with no forward and no ranking, in every path and backend. New decode
   paths ship with a `forwards_count == 0` bypass test or they do not merge.
6. **Speculate from forward-calculated symbol tables:** rank the legal domain
   with a deterministic scorer (`dsl/grammar/fastpath/speculative_rank.py`) and
   commit verified multi-token spans (lookahead-then-verify, arXiv:2602.00612;
   intersection witnesses, arXiv:2508.10111). Technique is a lever;
   verification before commit is not.
7. **Symbol tables schedule compute** (`runtime/decode_schedule.py`): minimal
   forwards, prefill boundaries at grammar checkpoints, device-aware routing.
   Record the counters in `DecodeStats`; utilization claims are measured.
8. **Output is scaffolded grammar.** NL vocabulary is optional fluff — never
   load-bearing, never a ship blocker.
9. **Use-case ladder, in order:** AST-2-AST → grammar-2-AST → grammar+ops-2-AST
   → simplified-NL-2-AST → complex-NL-2-AST. `CERT_CAP*` gates stay.
10. **Calculator/solver enhanced with inference** — not a chat model.
11. **Encoder vocabulary reserves a compute-ops vocabulary shared with the
    decoder** — `dsl/ops_vocab.py`, derived from the live operator registries
    and exposed through one `shared_token_ids()` mapping. Grammar symbols layer
    on top; NL sits above and is optional.
12. **Multi-turn = CRDT event store** over the conversation AST, with
    copy/undo/redo; the AST artifact materializes the full history.
13. **Patch/diff outputs across turns** wherever the edit space reaches the
    target. Full-AST output is the bootstrap mode, not the end state.
14. **Goals are non-negotiable; approaches are disposable.** A rejected
    experiment closes an approach, never a goal, and must file its successor.
15. **Everything is documented.** Changing an invariant means editing
    `docs/design/decode-invariants.md`, bumping `decode.invariants` in
    `resources/versions.json`, and passing
    `python -m scripts.verify_decode_invariants`.
<!-- /decode-invariants -->
