# Autoresearch and autotraining harness

**Status:** pluggable researcher harness implemented; no live upstream researcher,
model, paid GPU, or provider run in this change. The committed researcher fixture
benchmark is wiring evidence, not a model-quality claim.

## Goal and boundary

The harness turns repository history, external research, and previous experiment
feedback into bounded, falsifiable training experiments. It does not create a new
trainer or lineage. Training still flows through the canonical data, model,
evaluation, AgentV, checkpoint, and promotion surfaces.

Research and proposal compilation are separate stages:

1. a swappable `Researcher` implementation receives a bounded `ResearchRequest`
   and returns a cited memo, normalized sources, trajectory, and telemetry in a
   `ResearcherRun`;
2. the shared proposal compiler treats that memo as untrusted evidence and produces
   one strict `ExperimentSpec`; normal validation then enforces citations, campaign
   identity, allowlisted knobs, experiment budget, and the RL lock.

The registry in `src/slm_training/autoresearch/researchers.py` initially provides
two invocation adapters. Both run in a separately installed upstream checkout and
Python environment; no upstream package or dependency graph is vendored into this
repository.

| Researcher ID | Upstream entry point | Reviewed revision |
| --- | --- | --- |
| `open-deep-research` | LangGraph `deep_researcher.ainvoke` from [Open Deep Research](https://github.com/langchain-ai/open_deep_research) | `b764481fca7f0dbf00b2c70239bd97cea59d1059` |
| `open-researcher` | `deploy_agent.run_one` from [OpenResearcher](https://github.com/TIGER-AI-Lab/OpenResearcher) | `785fd6ba5fcbc068daa4a2f07bbe0964f2983c86` |

The runner refuses a checkout whose `git rev-parse HEAD` differs from the registry
pin, uses argv-only subprocess execution with a wall timeout and a 2 MB result
limit, and persists log hashes rather than log contents. Typed configuration
forbids unknown fields, so credentials remain environment variables in the isolated
process. Open Deep Research is MIT-licensed; the reviewed OpenResearcher checkout
does not contain an explicit license file, so do not redistribute it without a
separate license review.

Agent-authored and deterministic fixture proposals remain supported. The legacy
two-pass OpenAI provider remains for compatibility; new external-researcher runs
use `--compiler openai`, whose default is `gpt-5.6-sol` with `store=False`.

## Closed loop

```text
repo lineage + HF Daily Papers + web + prior artifacts
                         |
                         v
               immutable EvidenceSnapshot
                         |
                         v
       researcher -> cited memo/trajectory -> proposal compiler
                                                   |
                                                   v
                                      typed ExperimentSpec -> validation
                                                   |              |
                                          rejected |              | accepted
                                                   v              v
                                         researcher repair   compiled commands
                                                                  |
                                                                  v
                                         train data -> SFT/eval -> outcome
                                             ^                       |
                                             |                       v
                                      data repair <---- diagnosis ----> model repair
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
- `ResearchRequest`, backend-specific typed configs, and `ResearcherRun` record the
  exact upstream repository/revision, request hash, memo, normalized sources,
  trajectory, timing, and non-secret process telemetry;
- `ExperimentSpec` requires a hypothesis, expected effect, falsification and stop
  criteria, citations, parent, and typed `ExperimentKnobs`;
- `ExperimentOutcome` and `Diagnosis` route failures to data, researcher, model, or
  infrastructure remediation;
- `RLReadinessReport` is the only accepted RL capability token.

`compile_commands` constructs argv arrays from typed fields. No provider-authored
shell is evaluated. Embedded execution compiles the TwoTower and
`grammar_diffusion` data, training, and honest evaluation paths. Grammar campaigns
may vary only the allowlisted topology action, structural-embedding,
heterogeneous-noise, critic, buffer, and budget knobs. Causal-LM code or recipe
changes stay on the agent-driven `model_cycle` path so immutable parents and base
pins are preserved.

## Evidence and literature order

Evidence capture reads repository lineage first, then configured roots. The normal
root is `outputs/`, including lineage records, run summaries, raw telemetry,
AgentEvals/AgentV, annotation and preference feedback, data manifests and synthesis
telemetry, matrices, and previous autoresearch bundles.

Each completed train or performance-matrix run also emits
`outputs/runs/<id>/run_insights.json`. Its deterministic loss findings, phase
recommendations, and any persisted browser/OpenAI hypotheses are classified as
`run_insight` evidence and prioritized ahead of bulk output artifacts in the
bounded evidence snapshot. Generated suggestions therefore inform later proposal
compilation, but never enqueue an experiment directly; the normal typed-spec,
citation, budget, validation, and RL-lock checks still apply.

After local capture, `research` reads recent HF Daily Papers (`/api/daily_papers`)
and targeted historical paper search (`/api/papers/search`). A selected researcher
receives those sources and the immutable local evidence summary, and may discover
additional URLs. Each proposed citation must resolve to captured evidence or a
normalized captured source.

Committed source inventories can be added without network access:

```bash
python -m scripts.autoresearch research --campaign-id <id> --offline \
  --source-manifest src/slm_training/resources/autoresearch/dynamic-symbol-sources.json
```

`--source-manifest` is repeatable. Each file is validated as strict
`ResearchSource` records and merged by canonical URI before persistence; offline
mode skips HF/network discovery but still loads these reviewed sources.

## Persistence and observability

```text
outputs/autoresearch/<campaign>/
  campaign.json
  events.jsonl             # append-only hash chain
  results.tsv              # human-scannable event ledger
  checksums.jsonl
  artifacts/
    researcher_runs/<content-sha>.json   # pin + memo + trajectory + telemetry
    research_sources/<content-sha>.json  # normalized citation-valid source set
    experiments/<content-sha>.json       # compiler output after validation
  runs/<experiment>/...
```

Local artifacts are authoritative. Trackio is an optional live mirror. A complete
campaign can be mirrored to
`hf://buckets/TKendrick/OpenUI/autoresearch/<campaign>/`. `sync` is dry-run unless
`--push` is supplied. Full checkpoint and model-card rules still apply separately.
Run insight enrichment is stored in the run-local `run_insights.json` with its
source fingerprint and provider/runtime metadata. If source metrics change, stale
enrichment is rejected rather than attached to new evidence. A browser result that
cannot be persisted may be shown for the current UI session but is not autoresearch
evidence until the action endpoint writes it successfully.

## Isolated researcher setup

Installation is deliberately manual because the upstream environments are large,
networked, and provider-specific. Clone outside this repository, check out the
reviewed revision, and follow the upstream environment instructions. For example:

```bash
# Open Deep Research: upstream documents Python 3.11 + uv sync.
git clone https://github.com/langchain-ai/open_deep_research /path/open_deep_research
git -C /path/open_deep_research checkout b764481fca7f0dbf00b2c70239bd97cea59d1059
cd /path/open_deep_research && uv venv --python 3.11 && uv sync

# OpenResearcher: upstream documents Python 3.12 and a GPU/search-heavy stack.
git clone https://github.com/TIGER-AI-Lab/OpenResearcher /path/OpenResearcher
git -C /path/OpenResearcher checkout 785fd6ba5fcbc068daa4a2f07bbe0964f2983c86
cd /path/OpenResearcher && uv venv --python 3.12 && uv pip install -e .
```

Keep provider/search credentials only in that environment. A config file selects
non-secret options. Open Deep Research accepts `{}` for its reviewed defaults. A
minimal OpenResearcher config is:

```json
{
  "base_url": "http://127.0.0.1:8001/v1",
  "model": "OpenResearcher/OpenResearcher-30B-A3B",
  "browser_backend": "serper",
  "max_rounds": 50
}
```

The upstream OpenResearcher recipe describes an eight-A100 environment; adopting
its invocation surface here does not claim that local hardware can serve its model.

## Paper reproduction consideration

alphaXiv exposes an optional Autoresearch page at
`https://www.alphaxiv.org/replicate/<arxiv-id>`. Its current local-harness flow
installs and starts `orx`, then accepts a command of the form
`/reproduce-paper <arxiv-id> <title>`. This harness does **not** install it, submit a
reproduction, authenticate, allocate cloud compute, or duplicate a paper by
default.

An agent considering reproduction must:

1. ask for explicit approval before installation, authentication, cloud/GPU use,
   or cloning generated code into this repository;
2. work in a separate scratch repository or worktree and pin the paper version,
   author-code revision, dependencies, datasets, and seeds;
3. define a finite matrix of paper claims and acceptance rules before execution;
4. keep supplied paper assets separate from generated outputs and record provenance
   for every comparison;
5. treat a partial or failed reproduction as a measured result, not validation of
   the paper or a reason to weaken this repository's ship gates;
6. import only reviewed, minimal findings through the normal research → typed spec
   → experiment path, with the required JSON, markdown, AgentV, and model-card
   evidence for any runs or checkpoints produced here.

For the papers tracked in [`research-lineage.md`](research-lineage.md), this change
records applicability only. No alphaXiv reproduction or upstream training run was
started.

## Data iteration

Autoresearch data builds use a unique version and `--immutable`. Existing corpora can
become deterministic roots:

```bash
python -m scripts.build_train_data \
  --source existing \
  --derive-from outputs/data/train/<old>/records.jsonl \
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
| Benchmark / rerun | `researcher-d2585593fa0d3fee`; 2026-07-14 CDT |
| Researcher | `harness-v2` over deterministic fixture predictions |
| Backend / device / steps | fixture JSONL + pinned AgentV SDK / CPU / no training |
| Frozen set | 3 cases; data-family repair, mixture regression, weak-SFT RL lock |
| Grounded / valid / novel / actionable | `1.00 / 1.00 / 1.00 / 1.00` |
| AgentV | `3/3` pass, 0 execution errors, mean score `1.0`, 20 ms |
| Threshold | `0.80` per rate; benchmark pass |
| Promotion | **No** — `human_approved=false`, therefore `promotable=false` |
| Honesty | wiring-only researcher benchmark; no model/data quality or ship claim |

The measured command set `PYTHONPATH=src:.` and wrote its AgentV bundle under
`outputs/autoresearch/researcher_eval_fixture_v2/`. No provider, network, model
training, checkpoint, or GPU was used.

## RL is fail-closed

All GRPO-lite, trajectory RL, phase pipeline, quality/grammar matrix RL, NeMo RL,
and Molt RL entrypoints call the same library assertion. There is no override. One evaluation
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
For NeMo and Molt, the validated report is embedded and revalidated inside the
container before framework imports or optimizer work.

## Campaign commands

```bash
python -m scripts.autoresearch init \
  --campaign-id openui-sft-001 \
  --objective "Improve minimum-suite structure without parse regression" \
  --primary-metric min_suite.structural_similarity \
  --researcher-mode open-deep-research

# Topology campaigns use the same evidence/compiler boundary.
python -m scripts.autoresearch init \
  --campaign-id openui-topology-001 \
  --track grammar_diffusion \
  --objective "Improve honest topology composite without a ship-gate regression" \
  --primary-metric topology_composite

# The campaign researcher-mode selects the registry entry; an explicit
# --researcher overrides it. The runner verifies the exact reviewed Git pin.
python -m scripts.autoresearch research \
  --campaign-id openui-sft-001 \
  --researcher-checkout /path/open_deep_research \
  --researcher-python /path/open_deep_research/.venv/bin/python \
  --researcher-config open-deep-research.json

python -m scripts.autoresearch propose \
  --campaign-id openui-sft-001 --compiler openai

# Swap the researcher without changing the evidence, memo, or compiler contracts.
python -m scripts.autoresearch research \
  --campaign-id openui-sft-001 \
  --researcher open-researcher \
  --researcher-checkout /path/OpenResearcher \
  --researcher-python /path/OpenResearcher/.venv/bin/python \
  --researcher-config open-researcher.json

# Agent-authored specs remain a non-provider path.
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
