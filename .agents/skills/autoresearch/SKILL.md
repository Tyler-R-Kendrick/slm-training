---
name: autoresearch
description: Run the knowledge-driven OpenUI research loop that coordinates the training pipeline with curated knowledge. Read and update the repo + personal brains (OKF / Obsidian) and OpenWiki, run the prior-work discovery loop to find related research, drive the autotrain hypothesis loop to generate falsifiable hypotheses, and file new ideas and experiments as Linear issues / milestones / projects. Use to ORCHESTRATE research; to RUN a pipeline phase use autotrain; to RUN an experiment campaign (init/research/validate/execute) use openui-autoresearch; to change harness code use improve-openui-harnesses.
---

# Autoresearch — the OpenUI research loop

Facade for **orchestrating research** with progressive disclosure: this file
routes; each stage's full instructions live in `references/` and are read only
when that stage runs. This skill *coordinates* the machinery it does not own —
it drives `autotrain` (pipeline + hypothesis loop), `openui-autoresearch`
(campaign execution), the `openwiki` CLI, the brains under `docs/brains/`, and
the Linear MCP. It never builds a parallel trainer, evaluator, or knowledge store.

## What this loop does

Turns curated knowledge into falsifiable, tracked experiments and folds results
back into knowledge:

```
brains + OpenWiki  ──▶  discovery (prior work)  ──▶  hypotheses  ──▶  Linear issues
      ▲                                                                     │
      └──────────────  update brains from measured results  ◀──────────────┘
```

## Stage routing

Orient once per session with [references/loop.md](references/loop.md) +
[references/contracts.md](references/contracts.md) — `loop.md` owns the full
ordered sequence. The stages below run in table order; read a reference only
when its stage runs.

| # | Stage | Reads / writes | Reference |
| --- | --- | --- | --- |
| 1 | Read/update repo + personal brains (OKF / Obsidian) | `docs/brains/` | [references/brains.md](references/brains.md) |
| 2 | Read / refresh codebase knowledge | `docs/openwiki/` via the `openwiki` CLI | [references/openwiki.md](references/openwiki.md) |
| 3 | Find prior work & related research | `research-lineage.md`, source manifests, campaign bundle | [references/discovery.md](references/discovery.md) |
| 4 | Drive the auto-training hypothesis loop | `hypothesize` (owned by `openui-autoresearch` / `autotrain`) | [references/hypothesis.md](references/hypothesis.md) |
| 5 | File ideas/experiments as Linear issues/milestones/projects | Linear MCP, team `SLM` | [references/linear.md](references/linear.md) |
| 6 | Close the loop: fold results back into brains | `docs/design/` → `docs/brains/`, `synthesis-feedback` | [references/loop.md](references/loop.md) |

## Non-negotiable contracts

Digest — full versions in [references/contracts.md](references/contracts.md):

- **Knowledge, not results, lives in brains.** Measured results go to
  `docs/design/` (`documenting-experiment-results`); brains link them.
- **Grounded & novelty-audited.** Every hypothesis links prior work and is
  checked against recorded dead-ends before it is filed.
- **One idea → one issue → one `iter-*`.** A filed experiment maps to a Linear
  `SLM-N` issue and, on run, a `docs/design/iter-*.md` record.
- **Inherit the pipeline's laws.** Honesty (fixture vs ship), RL fail-closed,
  no shadow paths, and all approvals carry over from `autotrain`.
- **No secrets / no leakage.** Brains and issues never contain credentials,
  machine-absolute paths, or held-out eval content.
