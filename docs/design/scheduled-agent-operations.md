# Scheduled-agent operations

Runbook for the recurring maintenance work that used to run (or would
naturally run) on hosted CI schedules, now executed by **scheduled agent
sessions** instead — Claude Code Routines, or any scheduler the repository
owner attaches on their side.

## Operating model

- The repository commits **only the skills** that describe each operation
  (`.agents/skills/`, mirrored as symlinks into `.claude/skills/`,
  `.cursor/skills/`, `.grok/skills/`). It never commits scheduler
  configuration, scheduler credentials, provider API keys, or paid-service
  configs. Whoever owns the schedule holds those outside the repo.
- **Zero GitHub Actions minutes.** No scheduled workflows, no new workflow
  triggers, and no re-enabling of the disabled schedules — see
  [`ci-minutes-and-speed-plan-20260806.md`](ci-minutes-and-speed-plan-20260806.md).
  `.github/workflows/openwiki-update.yml` stays `workflow_dispatch`-only; the
  agent skill replaces its scheduled behavior.
- A scheduled session is an ordinary agent session: all repository laws apply
  unchanged (run cap, docs-follow-experiments, honest ship gates, never
  weakening gates or bypassing `scripts.verify_merge_ready`).
- Secrets (e.g. `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, Linear/GitHub auth)
  arrive only through the scheduled session's environment. A missing secret
  makes the affected operation **skip with a clear message**, never fail and
  never prompt committing the secret.

## Operations and suggested cadences

| Skill | Suggested cadence | What it does | Writes |
| --- | --- | --- | --- |
| [`openwiki-refresh`](../../.agents/skills/openwiki-refresh/SKILL.md) | **Weekly** | Install pinned `openwiki@0.1.2`, select provider (`OPENAI_API_KEY` → gpt, else `OPENROUTER_API_KEY` → glm, else SKIP), run `python -m scripts.update_openwiki --update --print`, PR the regenerated `docs/openwiki/` pages on branch `openwiki/update` | PR only (`docs/openwiki/`) |
| [`evidence-brief`](../../.agents/skills/evidence-brief/SKILL.md) | **Daily** | Read-only digest of autotrain evidence (climb policy, evidence ledger, closed approaches, evidence store, model card) posted to chat | Nothing |
| [`drain-in-review`](../../.agents/skills/drain-in-review/SKILL.md) | **On demand** (when the Linear `In Review` queue backs up) | Claim oldest team-SLM In-Review issues, one isolated worktree each, reuse open PRs, bounded CI-fix rounds, squash-merge only after merge preflight passes | PRs / merges via the normal gated path |

Cadences are suggestions for the scheduler owner; nothing in the repo enforces
them. Drain is deliberately not put on a timer — an unattended merge loop
should be started by a human noticing queue depth, not by a clock.

## Wiring a schedule (owner-side, outside the repo)

Example with Claude Code Routines: create a routine whose prompt is simply
"Use the `openwiki-refresh` skill in `/home/user/slm-training` and follow it
exactly", weekly cadence; likewise a daily routine for `evidence-brief`.
Any other scheduler (cron + CLI agent, a hosted agent platform) works the same
way: the schedule invokes an agent session that loads the committed skill.
The routine/cron definition and its credentials are the owner's — do not check
them in, and do not add repository files that embed them.

## Failure posture

- **Missing credential/provider** → operation-level SKIP with an explicit
  message in the session output.
- **Real failure** (script error, gate red, CI failure past bounded rounds) →
  report honestly; never weaken a gate, never bypass
  `scripts.verify_merge_ready`, never merge to go green.
- Anything a scheduled run measures or changes still follows the iron law:
  results are documented via `documenting-experiment-results`, and generated
  OpenWiki pages are never hand-edited.
