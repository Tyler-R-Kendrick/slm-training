---
name: phase-autoresearch
description: Bounded self-improvement — evidence-grounded autoresearch campaigns (init, research, hypothesize, validate, run, diagnose, sync), researcher/hypothesizer self-evaluation, and the validate-rl gate that produces the RLReadinessReport. Use when running autonomous experiment selection or campaign loops.
---

# Autoresearch (self-improvement) phase

Bounded self-improvement by accumulated evidence and policy iteration
(`docs/design/autoresearch-autotraining.md`). Owner:
`src/slm_training/autoresearch/`. Campaign methodology, provider rules, and
evidence contracts: **`openui-autoresearch`** — follow it alongside this phase.

## Prerequisites

- A falsifiable objective + budget; provider credentials only when using the
  OpenAI research path. Campaign bundle root:
  `outputs/autoresearch/<campaign>/` (canonical; Trackio/HF are mirrors).

## Commands

```bash
python -m scripts.autoresearch init --campaign-id <id> \
  --objective "<falsifiable objective>" --primary-metric <metric>
python -m scripts.autoresearch research --campaign-id <id>
python -m scripts.autoresearch hypothesize --campaign-id <id> --provider agent|openai
python -m scripts.autoresearch validate --campaign-id <id>
python -m scripts.autoresearch run --campaign-id <id>            # plan first
python -m scripts.autoresearch run --campaign-id <id> --execute  # typed commands only
python -m scripts.autoresearch diagnose --campaign-id <id>
python -m scripts.autoresearch status --campaign-id <id>
python -m scripts.autoresearch sync --campaign-id <id> [--push]

# Self-evaluation (frozen benchmarks; human promotion only)
python -m scripts.autoresearch evaluate-researcher ...
python -m scripts.autoresearch evaluate-hypothesizer ...

# RL gate: produces the approved report train_rl requires
python -m scripts.autoresearch validate-rl --evaluation <bundle> --output <report.json>

python -m scripts.autoresearch propose|materialize-mixture ...
```

## Gates & invariants

- Never accept researcher-authored shell/code; embedded experiments change only
  campaign-allowed `ExperimentKnobs`.
- Matrices need ≥5 grounded, novelty-audited candidates; completed experiments
  become typed hypothesizer feedback for the next matrix.
- The loop improves via evidence and typed feedback only — it never rewrites
  its own code, frozen cases, or acceptance thresholds.
- Self-improvement claims need frozen evaluation cases, held-out results, and
  explicit human promotion.
- No paid GPU, remote job, or HF write without explicit user approval.

## Close out

- Iron law docs per executed experiment (`documenting-experiment-results`);
  lineage in `docs/design/research-lineage.md`.
- Checks: `pytest -q tests/test_autoresearch` and
  `python -m scripts.autoresearch --help`.
