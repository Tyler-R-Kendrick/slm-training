# Autoresearch and autotraining harness

**Status:** harness implemented; no live model, paid GPU, or provider run in this
change. The committed researcher fixture benchmark is wiring evidence, not a model
quality claim.

## Goal and boundary

The harness turns repository history, external research, and previous experiment
feedback into bounded, falsifiable training experiments. It does not create a new
trainer or lineage. Training still flows through the canonical data, model,
evaluation, AgentV, checkpoint, and promotion surfaces.

There are two researcher modes:

- **agent-driven:** a coding agent may inspect and change code on an experiment
  branch, but its proposal must validate as an `ExperimentSpec` before entering the
  ledger;
- **embedded:** OpenAI Responses performs a web-search discovery pass followed by a
  Structured Outputs pass. It changes only typed, campaign-allowlisted knobs and
  cannot supply commands, Python, patches, or arbitrary configuration.

The embedded default is `gpt-5.6-sol`, with `store=False`. The local record keeps
requested and returned model IDs, response IDs, usage, prompt/evidence identity,
citations, and sources. It never stores API keys.

## Closed loop

```text
repo lineage + HF Daily Papers + web + prior artifacts
                         |
                         v
               immutable EvidenceSnapshot
                         |
                         v
      researcher -> typed ExperimentSpec -> validation
                         |                    |
                         | rejected           | accepted
                         v                    v
                  researcher feedback   compiled commands
                                              |
                                              v
                     train data -> SFT/eval -> outcome
                         ^                       |
                         |                       v
                  data repair <---- diagnosis ----> researcher repair
                                              |
                                              v
                  full competence + AgentV + reward variance
                                              |
                                              v
                                    RL readiness (locked)
```

Every arrow writes a content-addressed artifact and an append-only event.

## Schemas and safety

`src/slm_training/autoresearch/schemas.py` defines strict Pydantic models with
unknown fields forbidden:

- `CampaignSpec` and `CampaignBudget` fix objective, metric, track, evidence roots,
  allowed knobs, experiment count, wall time, and GPU-hour ceiling;
- `EvidenceSnapshot` records path, kind, content SHA, size, summary, and numeric
  metrics for lineage docs, run summaries, telemetry, AgentV, annotations, data
  manifests, matrices, and older campaigns;
- `ResearchSource` records HF Daily Paper, HF paper search, web, repository, or
  prior-run sources;
- `ExperimentSpec` requires a hypothesis, expected effect, falsification and stop
  criteria, citations, parent, and typed `ExperimentKnobs`;
- `ExperimentOutcome` and `Diagnosis` route failures to data, researcher, model, or
  infrastructure remediation;
- `RLReadinessReport` is the only accepted RL capability token.

`compile_commands` constructs argv arrays from typed fields. No provider-authored
shell is evaluated. Embedded execution currently compiles the TwoTower data,
training, and honest evaluation path. Causal-LM code or recipe changes stay on the
agent-driven `model_cycle` path so immutable parents and base pins are preserved.

## Evidence and literature order

Evidence capture reads repository lineage first, then configured roots. The normal
root is `outputs/`, including lineage records, run summaries, raw telemetry,
AgentEvals/AgentV, annotation and preference feedback, data manifests and synthesis
telemetry, matrices, and previous autoresearch bundles.

After local capture, `research` reads recent HF Daily Papers (`/api/daily_papers`)
and targeted historical paper search (`/api/papers/search`). The OpenAI discovery
pass performs general web search for remaining gaps. Each proposed citation must
resolve to captured evidence or a captured source.

## Persistence and observability

```text
outputs/autoresearch/<campaign>/
  campaign.json
  events.jsonl             # append-only hash chain
  results.tsv              # human-scannable event ledger
  checksums.jsonl
  artifacts/<kind>/<content-sha>.json
  runs/<experiment>/...
```

Local artifacts are authoritative. Trackio is an optional live mirror. A complete
campaign can be mirrored to
`hf://buckets/TKendrick/OpenUI/autoresearch/<campaign>/`. `sync` is dry-run unless
`--push` is supplied. Full checkpoint and model-card rules still apply separately.

## Data iteration

Autoresearch data builds use a unique version and `--immutable`. Existing corpora can
become deterministic roots:

```bash
python -m scripts.build_train_data \
  --source existing \
  --derive-from outputs/train_data/<old>/records.jsonl \
  --version <new> --immutable
```

The build emits `synthesis_telemetry.jsonl` with family counts, root-parent exposure,
quality ranges, and task counts, and includes its SHA in `manifest.json`. Existing
verifier and quality rejects, leakage fingerprints, governance files, mixture
diagnostics, cluster exposure, and content fingerprint remain authoritative.

When diagnosis targets data, change one filter, producer, or mixture lever; build a
new immutable snapshot; hold seed/token/evaluation snapshot constant; and compare a
matched control. Never edit a prior snapshot or train on the feedback eval holdout.

## Researcher improvement

Researcher changes are evaluated on
`src/slm_training/resources/autoresearch/researcher_cases.json`. The benchmark measures strict-spec
validity, grounded citations, distinct bounded knob signatures, and actionable
expected-knob/stop coverage. It publishes AgentEvals JSONL through the pinned AgentV
SDK. Promotion requires every score to clear the threshold, all cases to pass, and a
separate human approval. Frozen benchmark cases are evaluation-only.

### Measured fixture benchmark (2026-07-14 CDT)

Durable result: [`autoresearch-researcher-benchmark.json`](autoresearch-researcher-benchmark.json).

| Recipe / result | Value |
| --- | --- |
| Researcher | deterministic `fixture-v1` predictions |
| Backend / device / steps | fixture JSONL + pinned AgentV SDK / CPU / no training |
| Frozen set | 3 cases; data-family repair, mixture regression, weak-SFT RL lock |
| Grounded / valid / novel / actionable | `1.00 / 1.00 / 1.00 / 1.00` |
| AgentV | `3/3` pass, 0 execution errors, mean score `1.0` |
| Threshold | `0.80` per rate; benchmark pass |
| Promotion | **No** — `human_approved=false`, therefore `promotable=false` |
| Honesty | wiring-only researcher benchmark; no model/data quality or ship claim |

The first invocation failed before evaluation because the isolated worktree was not
on `PYTHONPATH`. The repeated command set `PYTHONPATH=src:.`; the measured AgentV run
then completed. No provider, network, model training, checkpoint, or GPU was used.

## RL is fail-closed

All GRPO-lite, trajectory RL, phase pipeline, quality/grammar matrix RL, and NeMo RL
entrypoints call the same library assertion. There is no override. One evaluation
bundle must prove:

1. `frozen_production_evaluation` metadata and a never-trained feedback holdout;
2. all five canonical suites;
3. full `rico_held` with `n >= 1500`;
4. unchanged canonical honest ship gates pass;
5. pinned AgentV evaluation passes;
6. at least two reward samples have nonzero variance.

```bash
python -m scripts.autoresearch validate-rl \
  --evaluation outputs/runs/<run>/rl_readiness_input.json \
  --output outputs/runs/<run>/rl_readiness.json
```

A failure is a supervised-model or data-improvement signal. Do not weaken the gate.
For NeMo, the validated report is embedded and revalidated inside the container
before imports or optimizer work.

## Campaign commands

```bash
python -m scripts.autoresearch init \
  --campaign-id openui-sft-001 \
  --objective "Improve minimum-suite structure without parse regression" \
  --primary-metric min_suite.structural_similarity

python -m scripts.autoresearch research --campaign-id openui-sft-001
python -m scripts.autoresearch propose \
  --campaign-id openui-sft-001 --provider openai
python -m scripts.autoresearch propose \
  --campaign-id openui-sft-001 --provider agent --proposal proposal.json
python -m scripts.autoresearch run \
  --campaign-id openui-sft-001 --experiment <artifact.json>
python -m scripts.autoresearch sync --campaign-id openui-sft-001
```

The run command is a dry plan unless `--execute` is supplied; sync is a dry plan
unless `--push` is supplied. No experiment was executed here. Future training,
evaluation, benchmark, profile, or decision-bearing telemetry is incomplete until
its JSON and matching markdown are committed under `docs/design/`.
