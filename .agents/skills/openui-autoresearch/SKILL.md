---
name: openui-autoresearch
description: Run or extend evidence-grounded OpenUI autoresearch and autotraining campaigns, including literature discovery, typed experiments, data repair, telemetry, researcher evaluation, persistence, and RL readiness.
---

# OpenUI autoresearch

Use this skill whenever work touches `scripts.autoresearch`, autonomous experiment
selection, research ingestion, campaign evidence, data-synthesis iteration, or RL
readiness.

## Non-negotiable contracts

- Read `AGENTS.md`, `docs/design/autoresearch-autotraining.md`, and
  `docs/design/research-lineage.md` first.
- Reuse the canonical lineage and data harnesses. Do not create a shadow trainer.
- Capture repository lineage and prior run evidence before outside research.
- Treat `outputs/autoresearch/<campaign>/` as the canonical raw bundle. Trackio and
  the HF Bucket are mirrors, not the source of truth.
- Never accept researcher-authored shell or code through the embedded provider.
  Embedded experiments may change only `ExperimentKnobs` fields allowed by the
  campaign.
- No paid GPU, remote job, or write to Hugging Face without explicit user approval.
- No train, eval, benchmark, telemetry, or reproduction run without the matching
  `docs/design/` JSON and markdown result. Use `documenting-experiment-results`.
- RL has no override. It requires an approved `RLReadinessReport` proving the frozen
  five-suite evaluation, full `rico_held`, honest ship gates, AgentV pass, and
  nonzero reward variance.

## Campaign loop

1. Initialize a budgeted campaign:

   ```bash
   python -m scripts.autoresearch init --campaign-id <id> \
     --objective "<falsifiable objective>" --primary-metric <metric>
   ```

2. Capture evidence and literature. Use `--offline` only for src/slm_training/resources/CI.

   ```bash
   python -m scripts.autoresearch research --campaign-id <id>
   ```

3. Propose through one of two paths:

   - `--provider agent --proposal <json>` for code-capable agent work.
   - `--provider openai` for the Responses web-search + Structured Outputs path.

4. Validate before execution. Citations must resolve to the captured evidence or
   source set, and every knob must be campaign-allowed.

5. Run without `--execute` first to inspect the compiled command plan. `--execute`
   runs only typed, locally compiled commands.

6. Persist the outcome and diagnose it. If data validity, leakage, or quality is
   bad, derive a new immutable snapshot with `--source existing --derive-from` and
   rerun matched controls. If a valid experiment repeatedly fails, improve the
   researcher policy and rerun the frozen researcher benchmark.

7. Sync only after the local bundle is complete:

   ```bash
   python -m scripts.autoresearch sync --campaign-id <id>       # dry plan
   python -m scripts.autoresearch sync --campaign-id <id> --push
   ```

## Researcher changes

Run the frozen fixture benchmark and publish AgentV evidence. A researcher is
promotable only when every score clears the threshold and a human explicitly
approves it. Never train the researcher on the frozen benchmark cases.

## RL readiness

Create the report from one complete evaluation bundle:

```bash
python -m scripts.autoresearch validate-rl \
  --evaluation <evaluation-bundle.json> \
  --output <rl-readiness.json>
```

A rejected report is useful evidence. Improve supervised competence or data first;
do not weaken the gate.
