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
slm autoresearch init --campaign-id <id> \
  --objective "<falsifiable objective>" --primary-metric <metric>
slm autoresearch research --campaign-id <id>
slm autoresearch hypothesize --campaign-id <id> --provider agent|openai
slm autoresearch validate --campaign-id <id>
slm autoresearch run --campaign-id <id>            # plan first
slm autoresearch run --campaign-id <id> --execute  # typed commands only
slm autoresearch diagnose --campaign-id <id>
slm autoresearch status --campaign-id <id>
slm autoresearch sync --campaign-id <id> [--push]

# Self-evaluation (frozen benchmarks; human promotion only)
slm autoresearch evaluate-researcher ...
slm autoresearch evaluate-hypothesizer ...

# RL gate: produces the approved report `slm rl train` requires
slm autoresearch validate-rl --evaluation <bundle> --output <report.json>

slm autoresearch propose|materialize-mixture ...
```

(`slm autoresearch <subcommand>` passes through to
`python -m scripts.autoresearch`.)

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
