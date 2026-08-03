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

Champion confirmation and promotion retain both source recipes. The queue stores
the candidate and its matched control knobs, while legacy entries recover the
control recipe from the source campaign. This is required for levers such as
training steps: applying the candidate step count to both arms would erase the
treatment and violate distinct-matrix validation. A crash before any
`experiment_started` event releases the reserved confirm/promote attempt; actual
measurements still consume the bounded attempt.

Confirmation steering is outcome-conditioned. Once the confirm arm finishes,
the handoff replaces preregistration-time claims with the observed disposition:
a re-held champion advances to promotion, while a rejected fingerprint is
exhausted and its quality/loss divergence becomes the next hypothesis signal.
The result matrix may nominate a non-exhausted runtime arm to keep the bounded
loop executable, but labels it diagnostic and separately prioritizes a new
size-matched quality-targeted objective. It never repeats the falsified
confirmation claim as next-run guidance.

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
A finalized AgentV decode timeout outranks the evaluator's generic non-zero process
exit when actions are routed: the handoff emits the canonical runtime repair first
and retains the content-bound `retry_measurement` behind it. A repair receipt can
therefore unblock, but never silently consume, the required frozen replay.

Frozen retries cross code updates through a governed successor, never by weakening
the current-main check. The successor copies the prior control/candidate recipe and
locked endpoints, arms, seeds, gates, and stopping rules; changes only campaign and
experiment identity plus the clean current source commit; and records
`replay_of_manifest_sha256`. Both arms must complete before the execution action is
acknowledged. A crashed campaign without a handoff remains append-only provenance
but cannot shadow the latest completed execution/retry authority. It remains in the
strict campaign lineage; an already-written gap is recovered only through one unique
chain of initialized-only campaigns, while completed or ambiguous gaps fail closed.
Replay arm identity is resolved by longest registered suffix, so hyphenated canonical
arms such as `component-plan` and `literal-close` cannot be truncated into an
unsupported last word.
Replay manifests carrying Lean formal obligations never inherit the source proof
digest. The successor is first materialized without obligations, runs and validates
a fresh current-campaign Lean preflight, then restores the frozen obligation
templates and policies under current campaign/experiment IDs bound only to the
new proof digest before hypothesis authorization or execution. An unproved or
timed-out preflight leaves both arms
unexecuted and fails closed.
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

The c1812 frozen-promotion replay exposed a formal-ordering gap: the loader saw
the source manifest's required Lean obligation and rejected the replay before a
successor campaign could create the mandated fresh proof. Campaign harness v110
allows the governed frozen recipe to load, strips the stale proof binding from
the current-main successor, regenerates and validates the Lean preflight, and
rebinds the unchanged formal claim policy before hypothesis authorization or arm
execution. This repairs orchestration only; it does not reuse proof evidence,
weaken the formal gate, or change the frozen model/eval recipes.
The same replay also exposed an identity gap before execution: governed
screening retries resolve registered arm suffixes, but promotion candidates use
the canonical `-promote` identity. Campaign harness v111 maps that identity onto the
matrix's authorized candidate slot before restoring the exact frozen experiment;
unknown non-promotion suffixes still fail closed.
Campaign harness v112 also copies the frozen experiment's formal claims alongside
its knobs and requires current-campaign candidate metrics before declaring a
control-only timeout replay terminal. A derived handoff produced by the earlier
bug is deterministically refreshed to `inconclusive` with the frozen retry still
queued; source-campaign candidate metrics cannot satisfy the current replay.
Campaign harness v113 closes the historical-replay edge case where the immediate
predecessor was itself created by the pre-v112 bug and therefore contains an empty
claim list. The loader recovers each claim only from its typed, content-bound formal
preflight artifact, verifies campaign, experiment, obligation, template, policy,
status, and recomputed obligation identity, then carries that exact claim into the
fresh-proof successor. Missing or inconsistent proof evidence fails closed.
Campaign harness v114 also corrects the successor binding itself: obligation IDs
are campaign- and experiment-scoped, so a replay must recompute them for the current
successor rather than copy the predecessor ID. The fresh preflight's recomputed ID,
the authorized experiment claim, and the successor manifest must now agree exactly.
After c1816 terminally reproduced a completion-loss runtime unblock at low absolute
meaningful-program quality, campaign harness v115 opens a distinct size-matched arm:
`component-edge-token`. The deterministic compiler marks non-root component positions
in the gold target, and the ordinary reconstruction CE is reweighted only at those
positions. The arm adds no parameters, detached head, decoder score, or legal authority;
it reports its own position count and mean CE for causal attribution.
The c1817 edge-token arm activated on its intended rows but reproduced the
control's meaningful-program outputs and missed the efficiency floor. Campaign
harness v116 therefore opens `component-edge-margin`: a distinct zero-parameter
objective over deterministic `component_bound` decisions. It trains the gold
child component to outrank the other compiler-legal component siblings by the
declared margin. The filter changes neither the candidate domain nor decode
authority, and its typed row telemetry makes an inactive objective fail visibly.
The completed c1819 frozen replay exposed a classification gap rather than a
model win: the candidate was faster enough to improve MPR/ms while halving MPR
and regressing both structural similarity and protected binder F1. Campaign
harness v117 forbids that ratio from overriding a regressed MPR, the role-owned
quality primary, or any required non-regression metric. Such cycles remain
useful runtime evidence but cannot enter the champion queue.
After v117 correctly exhausted c1819, c1820 stopped before experiment formation
because the quality-arm bank had no distinct successor. Campaign harness v118
opens `compiler-decision-token`: ordinary reconstruction CE is reweighted at
every gold position where the deterministic compiler exposes a legal branch.
This directly tests the observed coverage hypothesis—two or three component-edge
rows may be too sparse—without adding parameters, decoder scores, or legal
authority. Typed decision counts and mean CE make achieved density measurable.
Cycle c1821 activated that objective on 34 final-step decision rows, but both
three-document production batches exhausted one shared 24-second wall and left
all quality metrics unmeasured. Eval harness v78 makes the documented timeout
truly per record by scaling a batch wall with its record count, still capped by
the cumulative evaluator deadline. Campaign harness v119 also stops reserving
a promotion-only Lean execution lane during screening; promotion continues to
reserve and execute formal preflight. The exact c1821 arms replay before any
new model hypothesis.
The c1822 exact replay completes both arms under those repairs. The dense
decision-token arm improves smoke structure `.05237→.14623`, meaningful rate
`0→.6667`, component recall `.0833→.3333`, and p50 latency by 11.6%, with
parse, binder F1, and fidelity held. This is a three-document screening signal,
not ship evidence: equality remains zero and the unchanged absolute gates fail.
The recipe therefore enters fresh-seed confirmation, not promotion or RL.
Cycle c1823 repeats the same size-matched recipe at seed 101823 and falsifies
that screening effect: both arms score `.0575` structure, zero meaningful rate,
zero component recall, `.48889` binder F1, `.38889` fidelity, and zero reward;
the candidate is also 0.92% slower. Campaign harness v121 closes the queue
bookkeeping gap exposed by the replay path and opens `compiler-decision-margin`,
a zero-parameter stratified alignment objective over every compiler-decision
family. It optimizes only the gold-versus-legal-sibling ranking already consumed
by constrained decode; grammar authority and legal candidate domains are
unchanged.
Cycle c1824 validates the direct legal-choice signal but exposes its cost: the
all-family margin arm raises fixture structure `.13527→.4811`, MPR
`.3333→.6667`, recall `.1667→.4167`, binder F1 `.6333→.8222`, and fidelity
`.5278→.7222`, while emitted tokens rise `21→61`, forwards `4→15`, and p50
latency `973→3902` ms. The fixed quality-primary latency budget rejects the arm.
Campaign harness v122 preserves the margin recipe in both successor arms and
isolates deterministic completion bounds as the candidate treatment. It also
surfaces token, forward, prefill, and canvas costs in the terminal result table
so future quality/cost failures steer from their actual mechanism.
Cycle c1826 rejects that treatment on the strict compiler-tree path: bounded and
unbounded arms are identical on quality, 201 emitted tokens, 51 forwards, 28,928
prefill tokens, and 13,056 canvas tokens, while both completion-bound counters
remain zero. Campaign v124 therefore targets the observed 27.3-second compiler
cost with the existing completion-domain equivalence cache instead. It also
orders terminal headlines as quality, latency, tokens, forwards, compiler time,
and cache activity, keeping causal signals visible before cell truncation.

The c1811 pre-execution failure exposed a cadence-boundary gap: c1810 had
confirmed a champion, c1812 was the next protected promotion slot, and the
intervening screening bank was exhausted. The driver failed instead of waiting
usefully. Campaign harness v109 preserves the promotion cadence and held-out
suite boundary by spending that otherwise-empty slot on a second fresh-seed
confirmation of the same size-matched recipes. It never promotes early and
never recycles a rejected arm. See
[`autotrain-cycle-1811-promotion-wait-harness-failure.md`](autotrain-cycle-1811-promotion-wait-harness-failure.md).

The c1808 container-close screen is an exact quality null: both arms emit the
same 36 tokens with seven forwards and score `.3225` structural similarity,
while all meaningful, binder, component, fidelity, and reward metrics remain
zero. The loss did receive 116 eligible rows and drove margin violations from
`.60` on step one to zero on step two, but the matched control already chose
the legal closes; training wall time rose `3.07→9.35` seconds with no decode
benefit. Campaign harness v108 therefore tests the interaction that c1808 could
not identify: typed-family balance for the c1807 quality gains plus the same
container-close loss to prevent its runaway comma continuation. See
[`autotrain-cycle-1808-container-close-null.md`](autotrain-cycle-1808-container-close-null.md).

The c1807 typed-family balance screen improves structure, binder F1, recall,
and fidelity but produces no meaningful programs and increases p50 more than
5x. Decode telemetry attributes the cost to runaway legal continuation: 201
tokens and 51 forwards versus 27 and 5. Campaign harness v107 therefore adds a
grammar-derived `container-close` alignment filter that trains only gold `)` /
`]` choices competing with legal comma continuation. It changes training
ranking, never grammar legality or deterministic certification. See
[`autotrain-cycle-1807-typed-family-balance-rejected.md`](autotrain-cycle-1807-typed-family-balance-rejected.md).

The c1806 screen rejects direct grammar `STRUCT`-token weighting: it lowers
typed structure CE but regresses smoke structure and raises p50 more than 3x.
Attribution shows component CE worsens while the 61-token structure family
improves, exposing a count-imbalance tradeoff. Campaign harness v106 adds a
zero-parameter, count-normalized component/`STRUCT` family-mean auxiliary as
the distinct successor; legality and constrained decoding are unchanged. See
[`autotrain-cycle-1806-structure-token-rejected.md`](autotrain-cycle-1806-structure-token-rejected.md).

The c1805 exact frozen replay reproduced the c1804 control-only typed decode
timeout. Component-token weighting is therefore a runtime-specific unblock,
but its `.081733` structure and `7186.02` ms p50 fail absolute quality and
latency expectations. The arm is retired. Campaign harness v105 adds a
zero-parameter, size-matched `STRUCT`-token reconstruction successor with
typed loss attribution; it targets scaffold formation without changing the
grammar domain or constrained decoder. See
[`autotrain-cycle-1805-component-token-rejected.md`](autotrain-cycle-1805-component-token-rejected.md).

The c1804 component-token screen is incomplete because the matched control
timed out on all three smoke records. Candidate-only gains in component recall,
meaningful-program rate, binder F1, and fidelity are non-attributable; its low
structure and high latency remain warning signals. Per-family attribution shows
the objective reduced last-batch component CE `22.0968→17.5132`. The typed
handoff requires one exact frozen replay. See
[`autotrain-cycle-1804-component-token-incomplete.md`](autotrain-cycle-1804-component-token-incomplete.md).

The c1803 screening result rejects `ltr_prefix_loss_weight=1`: it ties every
smoke quality metric and worsens p50 by 174.21 ms. Campaign harness v104 adds
direct component-token reconstruction weighting plus per-step component,
prefix, and non-component CE/count attribution, targeting observed component
recall .16667 without changing capacity or constrained decode authority. See
[`autotrain-cycle-1803-scaffold-prefix-null.md`](autotrain-cycle-1803-scaffold-prefix-null.md).

The c1802 screening result rejects `design_md_dropout=.25`: against its
size-matched control it reduced smoke structure `.174167→.096400`, component
recall `.25→.0833`, and meaningful-program rate `.3333→0`. Campaign harness
v103 therefore preregisters zero-parameter prefix-LTR supervision as the next
distinct scaffold-learning hypothesis. See
[`autotrain-cycle-1802-design-dropout-rejected.md`](autotrain-cycle-1802-design-dropout-rejected.md).

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
After those registered quality families and their later binder/fidelity/alignment
successors are exhausted, the continuous bank advances to
`symbol_slot_augmentation=true`. That arm permutes request-local slots and
alpha-renames binders during training against an otherwise identical `false`
control. It is parameter-size matched, remains grammar constrained, and tests
opaque-symbol generalization without introducing surface-name features. If that
approach is rejected, `mask_pattern=mixed` is the next registered same-size
training-method arm against explicit `random`; it changes corruption exposure,
not deterministic decode legality. A null or rejected mixed-mask arm advances
to `symbol_boundary_loss_weight=1` against zero. That objective reweights the
existing output-token CE at opaque-symbol positions and their immediate
neighbors, adds no head or parameters, and does not affect the grammar oracle.
If boundary supervision is null, the next arm uses deterministic
`design_md_dropout=0.25` against zero to test scaffold-context reliance without
changing model size or constrained decode authority.

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
