# Autoresearch and autotraining harness

**Status:** pluggable researcher and feedback-driven five-candidate hypothesizer
harnesses implemented; no live upstream researcher, model, paid GPU, or provider run
in this change. The committed researcher fixture benchmark is wiring evidence, not a
model-quality claim.

## Goal and boundary

The harness turns repository history, external research, and previous experiment
feedback into bounded, falsifiable training experiments. It does not create a new
trainer or lineage. Training still flows through the canonical data, model,
evaluation, AgentV, checkpoint, and promotion surfaces.

Bare `/autotrain` uses an agent-supervised control plane rather than leaving a
multi-cycle subprocess in charge. The unbudgeted host goal syncs Git, runs one
bounded `--supervised` cycle, validates `AutotrainCycleHandoffV1`, executes its
typed harness/Lean/data/docs/SDLC actions, prints the compact matrix, and starts
the successor. `AutotrainLoopStateV1` supplies a durable heartbeat and resumable
phase. Fixture `climb_accepted` and full `ship_promoted` are intentionally
different verdicts.

The terminal dashboard calls a loop `RUNNING` only when the host can see the
driver process. An absent driver is `DEAD` (or preserves an explicit `BLOCKED`
state); stale persisted `RUNNING` text cannot override process truth.

Handoff coordination is enforced, not advisory. The supervisor records
content-bound `AutotrainActionReceiptV1` entries with `autoresearch ack-action`;
each receipt carries the SHA-256 and kind of every evidence item. Evidence is
rehashed when the receipt is written and whenever prerequisites are read; legacy
URI-only receipts remain readable history but cannot satisfy a prerequisite.
Action-specific checks require tracked documentation, campaign-bound data
artifacts, repair commits after the failed campaign, and delivery commits already
merged into `origin/main`.
The successor refuses to initialize while predecessor theorem-stop, harness, Lean,
data, docs, or delivery prerequisites remain unacknowledged. Execution/steering actions
(`retry_measurement`, `next_experiment`, `monitor`) remain part of the next-cycle
control flow rather than circular prerequisites.

Frozen retries cross code updates through a governed successor, never by weakening
the current-main check. The successor copies the prior control/candidate recipe and
locked endpoints, arms, seeds, gates, and stopping rules; changes only campaign and
experiment identity plus the clean current source commit; and records
`replay_of_manifest_sha256`. Both arms must complete before the execution action is
acknowledged. A crashed campaign without a handoff remains append-only provenance
but cannot shadow the latest completed execution/retry authority. It remains in the
strict campaign lineage; an already-written gap is recovered only through one unique
chain of initialized-only campaigns, while completed or ambiguous gaps fail closed.
Replay manifests carrying Lean formal obligations require a fresh formal preflight
and otherwise fail closed.
Lineage validation is shared by status and feedback traversal so those surfaces cannot
disagree about an initialized-only gap. When the newest handoff-less frozen replay
already has verified terminal events for both diagnostic matrix decision arms, the
next supervised run finishes status, Phase A classification, and its typed handoff
without executing either arm again. Partial execution, non-diagnostic promotion
campaigns, and stray artifacts do not qualify. Evaluation-stage
recovery uses the complete `scoreboard.json` envelope, preserving exit-8 honest
ship-gate rejection as completed model evidence rather than a harness failure.
The novelty and exhausted-knob guards authorize only replay arm IDs whose successor
manifest arms match the frozen source, whose normalized proposed knobs match the
frozen experiment, and whose source digest resolves in campaign lineage. This is a
content-bound retry exception, not permission to repeat an ordinary rejected recipe.

Continuous evidence discovery is predecessor-bounded: explicit `--evidence-root`
arguments replace the historical recursive `outputs/` default, and the driver
passes only the predecessor campaign, loop ledger, and SDLC ledger. Omitting the
flag retains the broad default for interactive compatibility.

Research and proposal compilation are separate stages:

1. a swappable `Researcher` implementation receives a bounded `ResearchRequest`
   and returns a cited memo, normalized sources, trajectory, and telemetry in a
   `ResearcherRun`;
2. the shared hypothesizer treats that memo as untrusted evidence and produces a
   strict `HypothesisMatrix` with at least five `ExperimentSpec` candidates plus one
   recommended experiment; normal validation enforces citations, campaign identity,
   distinct knob-value signatures, allowlisted knobs, experiment budget, and RL lock;
3. each terminal outcome and diagnosis becomes typed `HypothesisFeedback`. The next
   matrix must name its predecessor and acknowledge every supplied feedback ID, while
   older feedback is recaptured as prior-campaign evidence.

When a campaign preregisters `metric_expectations_sha256`, `autoresearch run`
can also replay a Lean `metric_certificate/v2` and attach typed
`OptimumFeedbackV1`. This is a cycle-level experiment signal, never a gradient
term. A theorem-backed miss stops the campaign and emits a separate
`repair_formal` prerequisite for the Lean owner. An assumption-backed miss
preserves the terminal outcome but blocks promotion and requires the successor
matrix to cover `measurement_control`, `training_method`, `architecture`,
`lean_model`, and `assumptions` with explicitly labeled candidates.

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
       researcher -> cited memo/trajectory -> hypothesizer (>=5 + recommendation)
                                                   |
                                                   v
                                  typed HypothesisMatrix -> validation
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
                                             ^              |
                                             |              v
                                  next matrix <--- typed hypothesis feedback
                                                                  |
                                                                  v
                                      full competence + AgentV + reward variance
                                                                  |
                                                                  v
                                                        RL readiness (locked)
```

Every arrow writes a content-addressed artifact and an append-only event.

Hill-climb progress is gated by claim class, locked held-out identity, multi-seed
primary LCB, exhausted-knob ledger, synthesis-feedback clearance before SFT, and
`EG_params` on capacity growth. Pure predicates live in
`src/slm_training/autoresearch/hillclimb.py`. Continuous-loop **volatile** knobs
(primaries, cadence, identity fields, recipe-null caps) are externalized in
`resources/experiments/autotrain_climb/policy.v1.json` via
`climb_policy.py` — see [`autotrain-climb-policy.md`](autotrain-climb-policy.md)
and the governance table in
[`experiment-campaign-governance.md`](experiment-campaign-governance.md#hill-climb-evidence-governance).

Promotion and ship claims further require **authoritative credit**: immutable
`observation_table` + locked `analysis_plan` + recomputed `credit_report`
artifacts. `credit_engine.compute_credit_report` owns paired effects, Holm rows,
and empirical promotability; structural campaign governance cannot clear a sole
`sufficient_evidence` failure. See
[`experiment-campaign-governance.md`](experiment-campaign-governance.md#authoritative-credit-promotion--ship)
and `resources/experiments/authoritative_credit/defaults.v1.json`.

## Schemas and safety

`src/slm_training/autoresearch/schemas.py` defines strict Pydantic models with
unknown fields forbidden:

- `CampaignSpec` and `CampaignBudget` fix objective, metric, track, evidence roots,
  allowed knobs, experiment count, wall time, and GPU-hour ceiling. The configurable
  `max_wall_minutes` defaults to and rejects values above the canonical
  `slm_training.levers.MAX_RUN_MINUTES`; it is one
  cumulative deadline shared by data build, training, and evaluation stages;
- `EvidenceSnapshot` records path, kind, content SHA, size, summary, and numeric
  metrics for lineage docs, run summaries, telemetry, AgentV, annotations, data
  manifests, matrices, and older campaigns;
- `ResearchSource` records HF Daily Paper, HF paper search, web, repository, or
  prior-run sources;
- `ResearchRequest`, backend-specific typed configs, and `ResearcherRun` record the
  exact upstream repository/revision, request hash, memo, normalized sources,
  trajectory, timing, and non-secret process telemetry;
- `ExperimentSpec` requires a hypothesis, expected effect, falsification and stop
  criteria, citations, parent, and typed `ExperimentKnobs`; optional
  `formal_claims` bind versioned Lean preflight templates with `required` or
  `advisory` policy;
- `HypothesisMatrix` requires at least five distinct candidates, a recommended
  member, selection rationale, and feedback/predecessor lineage when revising.
  Continuous matrices also carry ranked `NextRunPriorityV1` steering that cites
  its evidence and distinguishes Lean/reproduced-harness authority from speculative
  hypotheses. Each candidate records how research, prior traces, and prior results
  informed it plus a typed `CategoricalNoveltyAudit`;
- `HypothesisFeedback` records the tested hypothesis/signature, terminal metrics,
  diagnosis evidence, recommended actions, and optional hash-bound Lean optimum
  feedback without inventing causal support;
- `ExperimentOutcome` and `Diagnosis` route failures to data, researcher, model, or
  infrastructure remediation. Any partial scoreboard—timeouts, execution errors,
  explicit incomplete documents, or document counters that do not reconcile with
  explicit `document_n` (distinct from total rows `n`)—is
  measurement-incomplete infrastructure evidence, never a model quality result;
- `RLReadinessReport` is the only accepted RL capability token.

`compile_commands` constructs argv arrays from typed fields. No provider-authored
shell is evaluated. Embedded execution compiles the TwoTower and
`grammar_diffusion` data, training, and honest evaluation paths. Grammar campaigns
may vary only the allowlisted topology action, structural-embedding,
heterogeneous-noise, critic, buffer, and budget knobs. Causal-LM code or recipe
changes stay on the agent-driven `model_cycle` path so immutable parents and base
pins are preserved.

Embedded stages execute in a fresh process group. The canonical interrupt budget
sends `SIGINT` to the full tree, waits the canonical kill grace, then kills and
reaps the group if needed; stdout/stderr are disk-backed and only bounded tails are
retained. Typed stage results therefore come from complete stdout when available or
from the canonical `train_summary.json` / `scoreboard.json` artifact created or refreshed
by that exact stage. An unchanged artifact from an earlier attempt is rejected; a
bounded log tail is never treated as the authoritative result. A nominally
successful train is still incomplete unless its typed summary reports
`stopped_on=steps`, the requested step count, and a present checkpoint.
Conversely, `evaluate_model --ship-gates` exit 8 is a completed negative result only
when every suite declares `n`, `document_n`, completed/incomplete documents, and
decode timeouts; document counts reconcile with no incomplete or timed-out rows;
and the failed gate binds error-free AgentV summary/criteria plus existing
AgentEvals spec and result-index artifacts.

Structural screening arms couple learning to inference. Component-plan,
component-edge, component-inventory, binder-topology, and their joint arm set
both the preregistered auxiliary loss and its matching model-ranking decode
weight; the size-matched control prebuilds the same head with both weights zero.
This follows three complete loss-only nulls where targets and auxiliary learning
signals were present but the trained head was not directly consumed at decode.
The decode weight may rank only grammar-legal candidates and never weakens I6.
Champion fingerprints retain both weights through confirmation and promotion.

### Program experiments route through this loop (G1, SLM-46)

The DSL diffusion research program (tracks A-G) has no parallel ad-hoc loop:
its levers are allowlisted typed knobs (`asap_decode`, `decode_min_content`,
`denoiser_backend`, `bind_encoding`, `mask_pattern`) compiled to bounded
`scripts/train_model.py` flags, and
[`autoresearch/program_matrix.py`](../../src/slm_training/autoresearch/program_matrix.py)
encodes Track A as a `HypothesisMatrix` grounded in the committed evidence
trail (A1 diagnosis E248, A2 fixture row E259, the E3 literature manifest).
`tests/test_autoresearch/test_program_matrix.py` submits it through the
engine end-to-end — validation, bounded command compilation, and feedback
acknowledgement — with the hypothesizer-eval benchmarks untouched.

The verified-scope-solver campaign (VSS4-02, SLM-75) follows the same pattern on
the `grammar_diffusion` track:
[`autoresearch/verified_scope_matrix.py`](../../src/slm_training/autoresearch/verified_scope_matrix.py)
encodes a matched control plus the four VSS4-02 hypotheses (proof-checked exact
closure, dependency capsules vs lexical decomposition, cost-to-go energy ranking,
and late surface realization) as a `HypothesisMatrix` grounded in the committed
verified-scope-solver contract, the VSS4-02 fixture memo, and the fixture matrix
results. Its scope/topology knobs (`scope_contracts`, `scope_local_oracle`,
`scope_contract_negatives`, `topology_actions`) compile to bounded
`scripts/train_model.py` flags with `--ship-gates` always appended, and each
candidate carries the eight fail-closed correctness gates as falsification
criteria. `tests/test_autoresearch/test_verified_scope_matrix.py` submits it
through the engine end-to-end. The planning matrix does not replace the stable
E/X/P/Q/R matrices: after a candidate demonstrates a lever, the stable
verified-solver row, matched control, JSON, and markdown are registered through
the normal matrix workflow.

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

## Five-hypothesis gate and categorical novelty

Every campaign defaults to `min_hypotheses=5`; the value may be raised but not
lowered. `hypothesize` forms the matrix in one structured response, persists it as
a content-addressed artifact, and writes each member as an executable
`ExperimentSpec`. Candidates must differ by their complete knob/value signature,
not merely by prose or by the name of a changed field. A signature already present
in a finished campaign experiment is rejected. If captured research, prior-result,
or prior-trace evidence exists, the matrix must cite and explain use of each
available class. `run` fails closed unless its exact spec is a member of the latest
matrix with at least the campaign minimum.

This planning matrix does not replace the stable E/X/P/Q/R matrices. After a
candidate demonstrates a lever, register the stable ID, matched control, JSON, and
markdown through the normal matrix workflow.

The novelty audit is an **Adapted** engineering use of Wang and Buehler,
[*Self-Revising Discovery Systems for Science: A Categorical Framework for
Agentic Artificial Intelligence*](https://arxiv.org/abs/2606.01444) (2026):

- a regime schema types artifacts, operations, grammar, verifiers, and tools;
- fixed-regime iteration is search, not discovery;
- a verified transition maps the old schema into a new schema, preserves old
  artifacts and provenance, and transports them by left Kan extension;
- accepted post-transition artifacts outside the transported image are residual
  content. If no old operation reaches a new type, the empty comma category makes
  the transported fiber empty, so an accepted artifact there is forced residual;
- worthiness still needs a gate: paired evidence must pay for the new complexity
  (the paper uses MDL/AIC examples), survive stress tests, and preserve old accepted
  evidence or explicitly record recoding cost.

The implementation deliberately separates a **candidate audit** from a proof of
discovery. Before running an experiment there is no accepted post-transition state,
preservation map, or observed residual. Therefore `CategoricalNoveltyAudit.status`
can only be `candidate`. Its schema checks a declared regime extension has a truly
new element and that claimed residual elements are outside declared transport; it
also requires a reachability analysis, preservation checks, stress tests, and
worthiness criteria. This proves non-reachability only inside the declared finite
schema and assumptions. It cannot prove global literature novelty, scientific truth,
or SOTA. Those remain empirical claims requiring captured research, matched controls,
honest suites, and the repository's normal gates.

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
    hypothesis_matrices/<content-sha>.json # >=5 candidates + novelty audits
    formal_preflights/<content-sha>.json # proof/counterexample + source bindings
    hypothesizer_feedback/<content-sha>.json # terminal outcome + diagnosis lesson
    hypothesizer_telemetry/<content-sha>.json
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

## Standard hypothesis loop

`research` captures repository evidence and sources first. `hypothesize` uses a
completed isolated-researcher memo when one exists. If none exists, the OpenAI
hypothesizer performs its own `store=False` web-research pass before structured
matrix generation; discovered URLs join the captured source set and remain subject
to normal citation validation.

```bash
python -m scripts.autoresearch init --campaign-id <id> \
  --objective "<falsifiable objective>" --primary-metric <metric>
python -m scripts.autoresearch research --campaign-id <id>
python -m scripts.autoresearch hypothesize --campaign-id <id>
python -m scripts.autoresearch formalize --campaign-id <id>   # when claims exist
python -m scripts.autoresearch run --campaign-id <id>          # inspect recommendation
python -m scripts.autoresearch run --campaign-id <id> --execute
```

`run` defaults to the matrix recommendation but still accepts `--experiment` for an
exact matrix member. It cannot run a legacy standalone proposal. After execution it
persists the outcome, diagnosis, and feedback inside the same campaign tree. A later
`hypothesize` receives feedback from the latest matrix, must link that predecessor,
and cannot repeat any finished knob signature or campaign experiment ID. Campaign-wide
ID uniqueness prevents run budgets and outcome lineage from aliasing an older
candidate. Matrix candidates are reviewable plans; only uniquely started experiments
consume `max_experiments`.

Each campaign is bounded self-improvement by accumulated evidence and policy
iteration. Bare `/autotrain` may chain those bounded campaigns under one persistent
`loop_id`; it never creates an unbounded train process. `cycle_index`,
`predecessor_campaign_id`, `upstream_commit`, and `integration_commit` preserve
cross-cycle and merge provenance. The controller loads predecessor feedback across
campaign stores. Initialized-only cycles with a matrix but no terminal feedback are
skipped when selecting the newest earlier complete feedback context; the same state
on a completed cycle fails closed. The controller requires the locked manifest source
to equal the clean integrated commit and refuses a commit that does not contain the
fetched `origin/main`.
`autoresearch status --loop-id <id> --matrix --last 5` derives four between-run
tables from verified event chains: liveness, results (including measurement
completeness and diagnosis), diagnostic/harness/Lean signals, and ranked next-run
priorities (`--all` shows complete history).

An outcome may carry a typed `HarnessSignalV1`. Only a signal reproduced on the
frozen input can diagnose `target=harness`, and it must identify one canonical
harness family. The controller repairs that shared owner through
`improve-openui-harnesses`, then replays the identical model/data arm. This keeps
harness improvement attributable instead of allowing a model and judge to co-adapt.
An experiment outcome whose process status is `failed` is routed immediately to
`harness_failure` rather than being hidden inside a generic incomplete-measurement
retry. Frozen recipes rejected by the capability gate are immutable negative
evidence and are not replayed; the next cycle preregisters a valid successor.

Structural auxiliary screening arms couple each trained loss to its legal-candidate
decode ranker. Their lexer compiler-path companion is `compiler_decode_mode=tree`,
which is bound into both train and evaluation commands and copied to the
size-matched control. Confirmatory and promotion manifests retain the same compiler
mode. The compatibility validator remains fail-closed for any arm missing that
companion.

Lean optimum feedback crosses the same campaign boundary. A theorem-backed miss
stops the contradicted campaign and leaves the persistent outer goal in
measurement/formal-model repair. An assumption-backed miss blocks promotion and
requires all five `measurement_control`, `training_method`, `architecture`,
`lean_model`, and `assumptions` lanes in both the successor candidates and their
ranked priorities. The signal never becomes a gradient term or causal claim.

Ordinary researcher and hypothesizer changes promote locally when their frozen
benchmark passes. A change to an evaluator, metric, threshold, gate, or frozen case
requires a separate preregistered `ExperimentCampaignV1` meta-campaign with
unchanged held-out controls. It may not lower/delete a gate, train on frozen cases,
or change the meta-gate that judges itself.

Formal preflights are structural filters, not predicted experiment outcomes. A
locked `required` claim must have a fresh `proved` artifact; conditional,
refuted, unknown, missing, or source-drifted evidence blocks execution. Advisory
claims persist the same result without becoming an empirical quality gate. The
templates, fixed-point/history example, proof scopes, and concrete-to-abstract
trace contract are defined in
[`formal-autoresearch.md`](formal-autoresearch.md).

This is not online weight training or permission to silently edit frozen
evaluations, promotion policy, or ship gates. Hypothesizer implementation changes
must clear the separate preregistered held-out meta-campaign above before the
controller treats them as promoted behavior.

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

For an already published immutable corpus, set the typed `train_version` knob.
The compiler resolves it through the canonical `DataStore`, passes
`--train-version` to both train and evaluation, and automatically uses that
version's committed `mixture.json` unless the experiment supplies an explicit
typed mixture override. Matched TwoTower studies can also pin lexer/compositional
output, compiler-alignment weight and stratification, compiler decode mode,
schema/slot context, and DESIGN.md context through typed knobs; the compiler
passes the shared settings to train and evaluation rather than relying on CLI
defaults. Cache locality and checkpoint synchronization are typed as well;
CPU/scratch candidates must set `sync_checkpoints=false` explicitly when their
checkpoint is diagnostic-only. Compiled stages use the active Python interpreter;
an unavailable executable is persisted as a typed failed outcome instead of
escaping the campaign ledger.

Evaluation datasets are selected with the typed `eval_version` knob and resolved
through `DataStore`; campaign compilation must not assume a local `v1` directory.
Diagnostic fallback policy is typed too, so matched constrained evaluations can
set `allow_unconstrained_fallback=false` instead of inheriting permissive defaults.

## Researcher improvement

Researcher changes are evaluated on
`src/slm_training/resources/autoresearch/researcher_cases.json`. The benchmark measures strict-spec
validity, grounded citations, distinct bounded knob signatures, and actionable
expected-knob/stop coverage. It publishes AgentEvals JSONL through the pinned AgentV
SDK. Promotion requires every score to clear the automated frozen meta-gate and all
cases to pass. Frozen benchmark cases are evaluation-only; judge changes use the
separate meta-campaign described above.

Hypothesizer changes use the parallel frozen matrix evaluation in
`src/slm_training/resources/autoresearch/hypothesizer_cases.json`. It measures valid
five-candidate matrices, grounded evidence-role use, a regime-transition candidate
audit, an actionable recommendation, and exact feedback/predecessor lineage:

```bash
python -m scripts.autoresearch evaluate-hypothesizer \
  --predictions <hypothesis-matrices.jsonl> \
  --hypothesizer-id <policy-id> \
  --run-dir outputs/autoresearch/hypothesizer_eval/<policy-id> \
  --output docs/design/autoresearch-hypothesizer-benchmark.json
```

This publishes AgentV evidence. Passing still is not promotion: a human must approve
the policy, and every benchmark run must update matching JSON and measured-results
markdown. Never train or tune a hypothesizer on the frozen cases.

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
  --min-hypotheses 5 \
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

python -m scripts.autoresearch hypothesize \
  --campaign-id openui-sft-001 --provider openai

# Swap the researcher without changing the evidence, memo, or compiler contracts.
python -m scripts.autoresearch research \
  --campaign-id openui-sft-001 \
  --researcher open-researcher \
  --researcher-checkout /path/OpenResearcher \
  --researcher-python /path/OpenResearcher/.venv/bin/python \
  --researcher-config open-researcher.json

# Agent-authored matrices remain a non-provider path.
python -m scripts.autoresearch hypothesize \
  --campaign-id openui-sft-001 --provider agent --matrix hypothesis-matrix.json
python -m scripts.autoresearch run \
  --campaign-id openui-sft-001 --experiment <artifact.json>
python -m scripts.autoresearch sync --campaign-id openui-sft-001
```

The legacy single-spec `propose` command remains available for proposal inspection,
but it does not satisfy the matrix gate and cannot unlock `run`. The run command is
a dry plan unless `--execute` is supplied; sync is a dry plan
unless `--push` is supplied. No experiment was executed here. Future training,
evaluation, benchmark, profile, or decision-bearing telemetry is incomplete until
its JSON and matching markdown are committed under `docs/design/`.
