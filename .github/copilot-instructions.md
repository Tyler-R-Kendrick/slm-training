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


<!-- repo-laws -->
# Repository laws (identical for every agent)

Parity across harnesses is enforced by
`python -m scripts.verify_agent_surfaces`; see
[`docs/design/agent-harness-parity-audit.md`](../docs/design/agent-harness-parity-audit.md).

- **Hard run cap.** Every train / eval / bench / profile / matrix run and its
  supporting shell commands obey `MAX_RUN_MINUTES` in
  `src/slm_training/levers.py`. A timed out or killed run is never evidence.
- **Iron law: docs follow every experiment.** After any train / eval /
  benchmark / profile / telemetry / matrix / reproduction run, activate
  `documenting-experiment-results`. Numbers only in `outputs/`, chat, or a PR
  comment are incomplete work.
- **Honest ship gates.** Activate `honest-ship-eval` when evaluating, writing
  or interpreting gates, or claiming readiness. Say fixture-demo vs ship; never
  weaken a gate to green CI.
- **Data-quality law.** After any training-data build or synthesis, activate
  `synthesis-feedback` — read `quality_report.json`, `rejected.jsonl`, and
  `synthesis_feedback.json`, and fix the synthesis harness, never the gates.
- **Model card.** When a checkpoint is created, synced, bootstrapped, or
  promoted, update `docs/MODEL_CARD.md` **and** the README model-card summary.
- **Version stamps.** Results carry `version_stamp`; changing a watched
  metric / gate / harness / matrix file requires a component bump (or a
  `no-bump:` history note) in `src/slm_training/resources/versions.json` —
  `python -m scripts.verify_version_stamps --check` enforces it.
- **External test cases.** Agents edit mirrored JSON cases under
  `src/slm_training/resources/test_cases/` and refresh snapshots with
  `python -m scripts.refresh_test_cases <test-or-resource>`. Ordinary tests and
  CI stay read-only; run `refresh_test_cases --check --changed` before finishing.
- **Dashboard parity.** When you change a dashboard page
  (`src/apps/dashboard/src/pages/*.tsx`), keep its interpreted-mode
  `src/slm_training/web/static/openui/*.openui` program at parity and run
  `scripts/validate_page_dsl.py` — activate the `dashboard-openui-parity` skill.
- **Preregistered campaigns.** Experiment runners and promotion candidates use
  the `ExperimentCampaignV1` contract; never replace a locked confirmatory
  endpoint, arms, seeds, stopping rule, or gates after outcomes are visible.
- **Repository organization.** Activate `organize-repository` before creating,
  moving, renaming, or deleting tracked paths, and use `git mv` for every
  tracked relocation.
- **SDLC / multi-step delivery.** Multi-phase work uses the `sdlc` skill:
  subagents with incremental check-ins, official `gh stack` stacked PRs,
  bottom-up rubber-duck adversarial closeout (comments, CI, squash-merge),
  and Scalar/sparse/worktree workspaces.

Copilot CLI hooks in [`.github/hooks/`](hooks/) block raw `mv` of tracked paths
and run the changed-file checker after edits. Copilot Chat in the IDE gets no
hooks — run `python -m scripts.repo_policy` and `.githooks/check-changed`
yourself before finishing.
<!-- /repo-laws -->


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

Ids are the canonical `I*` ids from `docs/design/decode-invariants.md`. Cite
them, never a file-local number — every agent surface uses the same ids.

- **I6 — Never output invalid grammar.** Every production decode path is
  grammar-constrained end to end. Unconstrained arms
  (`--unconstrained-control`, HTTP `grammar_constrained=false`, eval `raw`
  arms) are diagnostic controls only — never production defaults, never
  serving paths, never shipped, certified, or gated on.
- **I6 — Fail closed.** Production configs must not set
  `allow_unconstrained_fallback=True`, must run finalize validation, and must
  raise rather than return uncertified text — in every backend, ONNX included.
  An empty legal domain is a dead end, never a full-vocabulary fallback. No
  load-bearing contract may silently widen when a dependency is unavailable.
- **I6 — No lever, experiment, or skill may weaken constrained decoding.**
  Levers may change *how legal symbols are chosen* — never *whether output is
  legal*. Weakening levers live in `levers.CONSTRAINT_WEAKENING_LEVERS` and are
  CI-blocked from production configs.
- **I1 — Deterministic completion paths bypass inference** wherever a
  deterministic answer exists. Deterministic decode proofs outrank learned,
  semantic, confidence, and preference scores.
- **I2 — Forced bypass on singletons:** exactly one valid next symbol means
  commit it with no forward and no ranking, in every path and backend. New
  decode paths ship with a `forwards_count == 0` bypass test or they do not
  merge.
- **I3 — Speculate from forward-calculated symbol tables:** rank the legal
  domain with a deterministic scorer
  (`dsl/grammar/fastpath/speculative_rank.py`) and commit verified multi-token
  spans (lookahead-then-verify, arXiv:2602.00612; intersection witnesses,
  arXiv:2508.10111). **I5 —** the technique is a lever; verification before
  commit is not.
- **I4 — Symbol tables schedule compute** (`runtime/decode_schedule.py`):
  minimal forwards, prefill boundaries at grammar checkpoints, device-aware
  routing. Record the counters in `DecodeStats`; utilization claims are
  measured.
- **I9 — Output is scaffolded grammar.** NL vocabulary is optional fluff —
  never load-bearing, never a ship blocker.
- **I10 — Use-case ladder, in order:** AST-2-AST → grammar-2-AST →
  grammar+ops-2-AST → simplified-NL-2-AST → complex-NL-2-AST. `CERT_CAP*`
  gates stay.
- **I10b — Calculator/solver enhanced with inference** — not a chat model.
- **I13 — Encoder vocabulary reserves a compute-ops vocabulary shared with the
  decoder** — `dsl/ops_vocab.py`, derived from the live operator registries and
  exposed through one `shared_token_ids()` mapping. Grammar symbols layer on
  top; NL sits above and is optional. The output tokenizer layout is likewise a
  frozen, checkpoint-bound contract
  (`resources/tokenizer_layout_registry.json`), never re-derived from whichever
  grammar backend happens to be live.
- **I11 — Multi-turn = CRDT event store** over the conversation AST, with
  copy/undo/redo; the AST artifact materializes the full history.
- **I12 — Patch/diff outputs across turns** wherever the edit space reaches the
  target. Full-AST output is the bootstrap mode, not the end state.
- **I14 — Goals are non-negotiable; approaches are disposable.** A rejected
  experiment closes an approach, never a goal, and must file its successor.
- **I7 — Every agent surface carries the law.**
  `python -m scripts.verify_agent_surfaces` certifies that every configured
  harness carries every repository law.
- **I15 — Everything is documented.** Changing an invariant means editing
  `docs/design/decode-invariants.md`, bumping `decode.invariants` in
  `resources/versions.json`, and passing
  `python -m scripts.verify_decode_invariants`.
<!-- /decode-invariants -->
