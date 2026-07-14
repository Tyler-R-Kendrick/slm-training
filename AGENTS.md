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
5. **Durable checkpoints** — real full HF-context trains upload checkpoints to
   the [OpenUI HF Bucket](https://huggingface.co/buckets/TKendrick/OpenUI)
   (`docs/design/checkpoint-bucket.md`).
6. **Model cards** — every new/promoted checkpoint updates
   [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) **and** the README “Model card
   (summary)” section.

Start: `README.md`, `docs/MODEL_CARD.md`, `docs/design/openui-twotower.md`,
`docs/design/quality-experiment-matrix.md`,
`docs/design/perf-experiment-matrix.md`, `docs/design/research-lineage.md`,
`docs/design/checkpoint-bucket.md`.

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
| `hf-cli` | Hub models/datasets/spaces, auth, cache, HF jobs, buckets, downloads |
| `huggingface-*` / `hf-*` / `trl-training` / … | Other [huggingface/skills](https://github.com/huggingface/skills) workflows (papers, datasets viewer, trainers, Spaces, memory estimate, …) |
| `playwright-cli` | Browser automation or playground e2e |

### Hugging Face skills + CLI

Source: [huggingface/skills](https://github.com/huggingface/skills) (Cursor:
marketplace installs `hf-cli`; additional skills via `hf skills add`).

Project install (already in `.agents/skills/`, symlinked for Cursor/Claude):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash   # once per machine
hf skills add --force                            # regenerate hf-cli
hf skills add <name> --force                     # one marketplace skill
hf skills update                                 # refresh installed marketplace skills
hf skills add --claude --force                   # Claude discovery symlinks
hf skills add --dest=.cursor/skills --force      # Cursor discovery (hf-cli)
```

Cursor also loads MCP from [`.cursor/mcp.json`](.cursor/mcp.json) (Playwright +
Hugging Face Hub MCP at `https://huggingface.co/mcp?login`). Optional UI
install: [Cursor marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).

Prefer `hf` over deprecated `huggingface-cli`. Auth: `hf auth login` /
`hf auth whoami`. CLI docs:
https://huggingface.co/docs/huggingface_hub/guides/cli

### Checkpoint bucket (full training runs)

**Bucket:** `hf://buckets/TKendrick/OpenUI` →
https://huggingface.co/buckets/TKendrick/OpenUI

| Run kind | Checkpoints |
| --- | --- |
| Full HF-context train (`train_model` / `remote_train`, default) | Sync to bucket under `checkpoints/<run_id>/` |
| Scratch matrix / CI / fixture demo | Local `outputs/` only (`--no-sync-checkpoints`) |

```bash
export HF_TOKEN=hf_...   # required for write; never commit
python -m scripts.train_model --train-dir outputs/train_data/v1 \
  --context-backend hf --run-id twotower_v1 --steps 200
# Manual / rescue sync:
python -m scripts.sync_checkpoints --run-dir outputs/runs/twotower_v1 --ensure-bucket
```

Agents must **not** treat a full HF train as done until
`train_summary.json` contains `checkpoint_bucket` with a successful remote URI
(or an explicit documented `--no-sync-checkpoints` / scratch reason). Use
`hf-cli` / bucket skills for inspection (`hf buckets list TKendrick/OpenUI -R`).

### Model card (required with every checkpoint)

Whenever a checkpoint is **created, synced, bootstrapped, or promoted**:

1. Update **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)** — roster row, eval table
   (suite `n` + metrics + pass/fail), recipe (device/steps/backend/honesty),
   bucket URI or local path, and append **Checkpoint history**.
2. Refresh **README → “Model card (summary)”** — short table only; link to the
   full card for detail. Do not let the README diverge from the card.
3. Keep claims honest (fixture / scratch matrix ≠ production HF ship).

Triggers include: `train_model`, `remote_train`, `bootstrap_playground`,
`sync_checkpoints`, matrix runs that designate a reusable champion, preference /
RL stages that write a new serving `*.pt`, and `--register-promoted`.

A checkpoint without a model-card + README summary update is incomplete work
(same bar as missing `docs/design/` measured-results).

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
5. If a checkpoint was written/promoted: update `docs/MODEL_CARD.md` **and**
   README “Model card (summary)”.
6. Commit docs with the experiment — no “docs later” TODO.

**Doc homes:** quality/ship → `quality-experiment-matrix.md` (+ adversarial
review on policy changes); perf → `perf-experiment-matrix.md` /
`runtime-performance.md`; checkpoints → `MODEL_CARD.md` + README summary +
`checkpoint-bucket.md`; lever-specific → that design doc.

| Excuse | Reality |
| --- | --- |
| "outputs/ is enough" | Reviewers read `docs/design/`. |
| "JSON written; markdown later" | Headline tables are the scoreboard. |
| "Failed/partial — skip docs" | Document failure + recipe. |
| "It's in the PR body" | PR text is ephemeral. |
| "Bucket URI is enough; skip the model card" | Card + README summary are how humans find the checkpoint. |

**REQUIRED SKILL:** `documenting-experiment-results`.

## Engineering norms

- Prefer harness/script changes over one-off notebooks.
- Preserve train/test isolation and structural leakage checks.
- Never reintroduce silent `gold.placeholders` channels under
  `honest_slot_contract=True`.
- Say fixture-demo vs ship. Do not weaken ship gates to green CI.
- Match existing style; no unrelated drive-by refactors.

```
docs/MODEL_CARD.md # checkpoint roster + eval (keep README summary in sync)
docs/design/       # matrices + measured results (source of truth)
scripts/           # train / eval / matrix / bench CLIs
src/slm_training/
.agents/skills/    # canonical skills for all tools
```
