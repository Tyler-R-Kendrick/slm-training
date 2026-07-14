# Agent instructions (all coding agents)

Applies to **every** coding agent (Cursor, Claude Code, Codex, Gemini, Copilot,
others). Prefer this file over tool-specific defaults on process conflicts.

## Repo goals

Experiment-first OpenUI layout SLMs:

1. **Honest models** — TwoTower / grammar-diffusion that clear multi-suite
   `--ship-gates`, not fixture memorizers (`docs/design/adversarial-review.md`).
2. **Measurable progress** — every train / eval / bench / matrix run leaves
   durable evidence under `docs/design/`.
3. **Research → code → results** — specs cite papers; harnesses implement
   levers; docs record what ran and whether gates passed.
4. **Ship vs demo** — fixture demos are wiring-only; production claims need full
   scoreboards (full `rico_held` / HF / DESIGN.md when claimed).

Start: `README.md`, `docs/design/openui-twotower.md`,
`docs/design/quality-experiment-matrix.md`,
`docs/design/perf-experiment-matrix.md`, `docs/design/research-lineage.md`.

## Skills

Canonical: **`.agents/skills/<name>/SKILL.md`**. Mirrored for discovery under
`.claude/skills/` and `.cursor/skills/` — keep them identical when editing
(repo-authored skills). Generated tooling skills may be symlinked instead.

**If a skill might apply (~1%), open and follow it before acting.**

| Skill | Use when |
| --- | --- |
| `documenting-experiment-results` | After any train / eval / bench / profile / matrix / telemetry run |
| `honest-ship-eval` | Eval, gates, readiness claims, metric changes, demo vs ship |
| `running-experiment-matrices` | Running or extending E* / X* / PQR / phase matrices |
| `hf-cli` | Hub models/datasets/spaces, auth, cache, HF jobs, buckets, downloads (context tower / RICO / checkpoints) |
| `playwright-cli` | Browser automation or playground e2e |

### Hugging Face CLI (`hf`)

Install the CLI (once per machine):  

`curl -LsSf https://hf.co/cli/install.sh | bash`

Regenerate / refresh the skill from the installed CLI:

```bash
hf skills add --force          # .agents/skills/hf-cli
hf skills add --claude --force # also symlink .claude/skills/hf-cli
# Cursor discovery (symlink into .cursor/skills):
hf skills add --dest=.cursor/skills --force
```

Prefer `hf` over deprecated `huggingface-cli`. Auth: `hf auth login` /
`hf auth whoami`. Docs: https://huggingface.co/docs/huggingface_hub/guides/cli

## Iron law: docs follow every experiment

```text
NO TRAIN / EVAL / BENCH / PROFILE / TELEMETRY / MATRIX / REPRO
WITHOUT UPDATING DOCS
```

Numbers only in `outputs/`, chat, or a PR comment = incomplete work.

**Triggers (complete):** `train_model`, `train_rl`, `train_preference`,
`remote_train`, `evaluate_model`, `evaluate_loss_suites`, `diagnose_eval`,
`run_quality_matrix`, `run_grammar_matrix`, `run_perf_matrix`,
`run_phase_pipeline`, `reproduce_baseline`, `run_scaling_ladder`,
`run_mixture_search`, `bench_*` (incl. telemetry/accel), `profile_generate`,
or any ad-hoc run whose scoreboard / gates / latency inform a decision.

**Required each time:**

1. JSON under `docs/design/` (scripts often mirror; verify it matches this run).
2. Matching markdown measured-results / notes updated (not JSON-only).
3. Recipe metadata: device, steps, backend, matrix set, suite `n`, honesty mode.
4. Honest pass/fail vs `--ship-gates` or perf guardrails.
5. Commit docs with the experiment — no “docs later” TODO.

**Doc homes:** quality/ship → `quality-experiment-matrix.md` (+ adversarial
review on policy changes); perf → `perf-experiment-matrix.md` /
`runtime-performance.md`; lever-specific → that design doc.

| Excuse | Reality |
| --- | --- |
| "outputs/ is enough" | Reviewers read `docs/design/`. |
| "JSON written; markdown later" | Headline tables are the scoreboard. |
| "Failed/partial — skip docs" | Document failure + recipe. |
| "It's in the PR body" | PR text is ephemeral. |

**REQUIRED SKILL:** `documenting-experiment-results`.

## Engineering norms

- Prefer harness/script changes over one-off notebooks.
- Preserve train/test isolation and structural leakage checks.
- Never reintroduce silent `gold.placeholders` channels under
  `honest_slot_contract=True`.
- Say fixture-demo vs ship. Do not weaken ship gates to green CI.
- Match existing style; no unrelated drive-by refactors.

```
docs/design/      # matrices + measured results (source of truth)
scripts/          # train / eval / matrix / bench CLIs
src/slm_training/
.agents/skills/   # canonical skills for all tools
```
