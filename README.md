# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), a **TwoTower** masked-diffusion model, plus a **GPU multi-farm MCP**.

> **This is not a natural-language LLM.** It is a **grammar-constrained symbolic
> diffusion model** that emits templated grammars — scaffolded structure and
> structural reasoning — with templated content deferred to a real external LLM.
> Constrained decoding is the product, not a switch: deterministic completion
> paths bypass inference, a scope-proven singleton is committed with no forward
> pass, and no lever or experiment may make output less legal. The goal law is
> [`AGENTS.md` § Non-negotiable architecture invariants](AGENTS.md) and its
> canonical expansion [decode-invariants.md](docs/design/decode-invariants.md).

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — lineage-first **TwoTower** and causal-LoRA tracks
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

Autonomous experiment campaigns use the fail-closed, evidence-grounded
[`autoresearch` harness](docs/design/autoresearch-autotraining.md), with isolated,
pinned [Open Deep Research](https://github.com/langchain-ai/open_deep_research) and
[OpenResearcher](https://github.com/TIGER-AI-Lab/OpenResearcher) implementations
behind one memo/trajectory contract and trusted hypothesizer. Before execution the
pipeline requires a persisted matrix of at least five distinct, grounded hypotheses,
including categorical candidate-novelty audits adapted from
[Wang and Buehler (2026)](https://arxiv.org/abs/2606.01444). Pre-run audits are not
claims of proven discovery or SOTA. Each matrix names its recommended experiment;
completed outcomes and diagnoses become typed feedback for the next matrix and for
future campaign evidence. The loop improves by evidence, never by rewriting its own
code, frozen cases, or gates. RL remains locked until a model passes the frozen
production readiness contract. Architecture hypotheses can additionally use
[Lean formal preflights](docs/design/formal-autoresearch.md) to reject structural
contradictions before training; these prove explicit abstractions and assumptions,
never empirical quality or ship readiness.

See [docs/design/decode-invariants.md](docs/design/decode-invariants.md) (goal law — constrained decoding, deterministic bypass, symbol-table speculation and scheduling, shared ops vocab, CRDT multi-turn), [docs/design/model-lineage.md](docs/design/model-lineage.md) (canonical two-track cycle), [docs/design/openui-twotower.md](docs/design/openui-twotower.md), [docs/design/grammar-topology-diffusion.md](docs/design/grammar-topology-diffusion.md) (dynamic production-tree diffusion), [docs/design/verified-scope-solver.md](docs/design/verified-scope-solver.md) (VSS0 verified scope-solver contract — prefix legality vs verified support), [docs/design/research-lineage.md](docs/design/research-lineage.md) (papers → code), [docs/design/semantic-planning-valid-state-disposition.md](docs/design/semantic-planning-valid-state-disposition.md) (SPV4-02 final disposition), [docs/design/recurrent-semantic-computation-looped-latent-disposition.md](docs/design/recurrent-semantic-computation-looped-latent-disposition.md) (RSC4 blocked; not ship), [docs/design/research-correction-critics.md](docs/design/research-correction-critics.md) (V4 remask / trust-gate / honest inventory; V6 CoRe/T2M), [docs/design/verifier-stack.md](docs/design/verifier-stack.md) (G0–G12 corpus gates + confidence tiers), [docs/design/abstraction-house-style.md](docs/design/abstraction-house-style.md) (L0–L5 determinacy, grounding, and canonical defaults), [docs/design/verifier-guided-repair.md](docs/design/verifier-guided-repair.md) (PDDL-Instruct / verifier-repair applicability map), [docs/design/quality-experiment-matrix.md](docs/design/quality-experiment-matrix.md) (E0–E75 + X0–X15 matrices; E34 deferred), [docs/design/speculative-denoising.md](docs/design/speculative-denoising.md) (V7 stability / dependency-cluster / survival / successor-cache decode), [docs/design/dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md) (V5 lexer alphabet), [docs/design/grammar-fastpath.md](docs/design/grammar-fastpath.md), [docs/design/grammar-backends.md](docs/design/grammar-backends.md), [docs/design/dsl-pack-contract.md](docs/design/dsl-pack-contract.md) (F1 DSL-pack contract; OpenUI first pack), [docs/design/structure-only-eval.md](docs/design/structure-only-eval.md), [docs/design/binding-aware-meaningful-v2.md](docs/design/binding-aware-meaningful-v2.md) (versioned binding-aware metric and gaming audit), [docs/design/judge-independence-audit.md](docs/design/judge-independence-audit.md) (EFS0-04 cross-family/human audit contract), [docs/design/adversarial-review.md](docs/design/adversarial-review.md), [docs/design/runtime-performance.md](docs/design/runtime-performance.md), [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md) (HF Jobs full train — not ZeroGPU), [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md), and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

Final spectral program policy:
[null-calibrated spectral learning disposition](docs/design/null-calibrated-spectral-learning-disposition.md)
(diagnostics only; no spectral training/default/promotion adoption).

Calculated arity, task rate, neural precision, and physical cost are kept distinct
by the [CAP0 contract](docs/design/calculated-arity-adaptive-precision.md).

## Model card (summary)

Full card: **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)**. Agents update both this
summary and the full card whenever a checkpoint is created or promoted.

**Current compatibility:** output contract v2 is symbol-only. All pre-E714
checkpoints are provenance-only; current code refuses to load, serve, resume,
promote, or evaluate them. E714 is the first compatible scratch baseline, but
it fails semantic gates and is not promoted. See
[the contract](docs/design/symbol-only-output-contract.md).

**Spectral disposition:** the final NCS audit adopts only fail-closed
diagnostics. It rejects or blocks raw-alpha quality claims, spectral training
controls, causal spectral retention, and WW-PGD/trace-log projection. No
checkpoint, roster, training default, champion, or promotion status changes.

| Role | Checkpoint | Where | Claim |
| --- | --- | --- | --- |
| Autotrain 8c0b60dd-c2 (2026-08-04) bounds/control quality-neutral replay | 2 size-matched 1,608,962-param scratch checkpoints (frozen replay of c1) | `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2/runs/` (local, explicit no-sync) | First cycle in this session's loop to complete train+eval end to end after the c1 infra repair; bounds vs control tie on smoke structure `.0575` both, bounds `125.75` ms faster p50; rejected, quality-neutral, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-20260804-c2-bounds-quality-neutral-replay.md)) |
| Autotrain 8c0b60dd-c1 (2026-08-04) AgentV npm-ci infra failure | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1/runs/` (local, explicit no-sync) | Both arms trained (loss `22.6219` each) but `--ship-gates` eval crashed again on a missing AgentV SDK (fresh container, `npm ci` never run) before any scoreboard; not scoreable, not a model result. Fixed with a cycle-start preflight (`_require_agentv_sdk_available`) plus a latent `design_md` bridge `NODE_OPTIONS` bug found in the same investigation; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-20260804-c1-agentv-npm-ci-infra-failure.md)) |
| Autotrain c2 component-plan screen | 2 size-matched 1,755,764-param CPU scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c2/runs/` (local, explicit no-sync) | Fixture structural win `.3267→.3828`, meaningful `0`, recall `.1667`, parse `1.0`; candidate p50 `18,272.11` vs `23,124.96` ms. Smoke `n=3`, full suites missing. **Fresh-seed confirmation queued; not promotable/synced/ship-eligible** ([results](docs/design/autotrain-cycle-20260803-c2-component-plan-screen.md)) |
| Autotrain c1 bounds screen | 2 size-matched 1,608,962-param CPU scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1/runs/` (local, explicit no-sync) | Quality null: structure `.0575`, meaningful `0`, binder F1 `.6333`, parse `1.0`; candidate p50 `981.72` vs `971.56` ms. Smoke `n=3`, full suites missing. **Rejected; no promote/sync/ship** ([results](docs/design/autotrain-cycle-20260803-c1-bounds-screen.md)) |
| Autotrain c1863 semantic-contrast/compiler-margin confirmation rejected | 2 size-matched scratch checkpoints (1,608,962 params each) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1863/runs/ (local, explicit no-sync) | Fresh matched replay: struct .1742→.2899, MPR .333 tie, exact 0, p50 909→4875 ms, tokens 30→137. **Rejected for cost; no ship claim** ([results](docs/design/autotrain-cycle-1863-semantic-contrast-compiler-margin-confirmation-rejected.md)) |
| Autotrain c1862 semantic-contrast/compiler-margin confirmation incomplete | Candidate scratch checkpoint (1,608,962 params); control had no scoreboard | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1862/runs/ (local, explicit no-sync) | Candidate-only smoke struct .4197, MPR .333, recall .167, exact AST/canonical 0; **inconclusive because matched control was interrupted** ([results](docs/design/autotrain-cycle-1862-semantic-contrast-compiler-margin-confirmation-incomplete.md)) |
| Autotrain c1861 semantic-contrast/compiler-margin frozen replay | Reused c1860 size-matched scratch checkpoints (1,608,962 params each) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1861/runs/ (local, explicit no-sync) | Complete matched fixture signal: struct .0575→.2742, MPR 0→.333, p50 16759→3626 ms, exact AST/canonical 0, smoke n=3. **Not promotable; full-suite evidence missing** ([results](docs/design/autotrain-cycle-1861-semantic-contrast-compiler-margin-replay.md)) |
| Autotrain c1860 semantic-contrast/compiler-margin incomplete | 2 size-matched scratch checkpoints (1,608,962 params each) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1860/runs/ (local, explicit no-sync) | Candidate smoke struct .2742, MPR .333, exact 0; control evaluation interrupted before scoreboard. Incomplete; no attribution/promote/sync/ship ([results](docs/design/autotrain-cycle-1860-semantic-contrast-compiler-margin-incomplete.md)) |
| Autotrain c1859 constraint-graph null | 2 size-matched scratch checkpoints (1,608,962 params each) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1859/runs/ (local, explicit no-sync) | Quality/work tie; candidate p50 2536→2623 ms, exact 0, smoke n=3. Rejected fixture null; new objective required ([results](docs/design/autotrain-cycle-1859-constraint-graph-null.md)) |
| Autotrain c1858 slot-contract context replay | 2 size-matched scratch checkpoints (1,608,962 params each) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1858/runs/ (local, explicit no-sync) | Exact replay is a quality null: all guarded quality and decode work tie; candidate p50 942→871 ms is efficiency-only on smoke n=3, so no champion ([results](docs/design/autotrain-cycle-1858-slot-contract-context-replay-null.md)) |
| Autotrain c1857 slot-contract context harness failure | 1 completed control checkpoint; candidate failed before scoreboard | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1857/runs/ (local, explicit no-sync) | Candidate hit a shared inventory-extraction bug from an incidental DESIGN.md `:slot_4` example; repaired and exact replay required, no model attribution ([results](docs/design/autotrain-cycle-1857-slot-contract-context-harness-failure.md)) |
| Autotrain c1855 exposure-cap fresh confirmation | 2 size-matched scratch checkpoints (1,608,962 vs 1,613,477 params) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1855/runs/ (local, explicit no-sync) | Fresh seed raises structure .230→.354 and MPR 0→.333, but recall .333, exact 0, p50 2433→2574 ms, loss 10.20→18.46; confirmation rejected, no champion ([results](docs/design/autotrain-cycle-1855-exposure-confirmation-null.md)) |
| Autotrain c1854 slot-component exposure cap | 2 size-matched scratch checkpoints (1,608,962 vs 1,613,477 params) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1854/runs/ (local, explicit no-sync) | Fixture structure .0575→.1353, MPR 0→.333, recall 0→.167; p50 917→1005 ms, tokens +71%, forwards +75%, exact 0, n=3. Fresh confirmation queued; not ship ([results](docs/design/autotrain-cycle-1854-slot-component-exposure-cap-positive.md)) |
| Autotrain c1853 slot-component/inventory coupling | 2 size-matched scratch checkpoints (1,608,962 vs 1,686,878 params) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1853/runs/ (local, explicit no-sync) | All guarded quality remains zero and structure ties at .115; candidate adds +77,916 params and p50 856→902 ms, exact 0, n=3; rejected capacity-negative null ([results](docs/design/autotrain-cycle-1853-slot-component-inventory-coupling-null.md)) |
| Autotrain c1852 slot-component/fidelity coupling | 2 size-matched scratch checkpoints (1,608,962 vs 1,613,477 params) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1852/runs/ (local, explicit no-sync) | Guarded quality is an exact tie (structure .1742, MPR .333, recall .25, binder .633, fidelity .528); candidate loss/params/p50 worsen (12.00→24.16, +4,515, 910→966 ms), exact 0, n=3; rejected null ([results](docs/design/autotrain-cycle-1852-slot-component-fidelity-coupling-null.md)) |
| Autotrain c1851 slot-component coverage | 2 size-matched scratch checkpoints (1,608,962 vs 1,613,482 params) | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1851/runs/ (local, explicit no-sync) | MPR 0→.667 and fidelity/binder remain 1.0, but structure .1425→.1767, p50 7400→8771 ms, params +4,520, exact 0, n=3; rejected fixture tradeoff ([results](docs/design/autotrain-cycle-1851-slot-component-coverage-null.md)) |
| Autotrain c1849 slot-owner runtime gap | 1 completed control checkpoint; candidate failed at model construction | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1849/runs/ (local, explicit no-sync) | Reserved binder-slot owner has no runtime implementation; candidate did not train. Replaced by implemented slot-component coverage owner; replay/new hypothesis required ([results](docs/design/autotrain-cycle-1849-slot-owner-runtime-gap.md)) |
| Autotrain c1848 binder-slot-ownership repair | 1 completed control checkpoint; candidate failed before training | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1848/runs/ (local, explicit no-sync) | Harness measurement incomplete: capability gate rejected the candidate because tree compiler mode was missing. Replay required after repair; no model attribution or ship claim ([results](docs/design/autotrain-cycle-1848-binder-slot-ownership-harness-repair.md)) |
| Autotrain c1847 semantic-exhaustive successor | 2 size-matched 1,608,962-param scratch checkpoints | outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1847/runs/ (local, explicit no-sync) | Structure .2383→.3225 and decode work/p50 improve, but binder .9524→.8222, fidelity .9167→.7222, MPR/exact metrics remain 0, and the fixture is n=3. Rejected ([results](docs/design/autotrain-cycle-1847-semantic-exhaustive-null.md)) |
| Autotrain c1846 exposure-targeted quality/cost | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1846/runs/` (local, explicit no-sync) | MPR `.3333→.6667`, recall `.25→.4167`, binder/fidelity `→1.0`, but structure -.052, tokens +117%, and p50 +138%. Quality signal retained; configuration rejected ([results](docs/design/autotrain-cycle-1846-exposure-quality-cost.md)) |
| Autotrain c1845 valid capacity-aware tail confirmation | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1845/runs/` (local, explicit no-sync) | Lever-complete fresh-seed candidate/control quality and work are identical; candidate p50 +1.6%. c1830 signal is seed-sensitive and rejected; never reuse/promote/sync/ship ([results](docs/design/autotrain-cycle-1845-capacity-tail-valid-confirmation-null.md)) |
| Autotrain c1844 promotion harness failure | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1844/runs/` (local, explicit no-sync) | Lean proved and trains completed, but 8/8 eval documents timed out; audit found inner-budget loss and champion recipe drift. No model attribution; never reuse/promote/sync/ship ([results](docs/design/autotrain-cycle-1844-promotion-recipe-and-budget-failure.md)) |
| Autotrain c1843 structure-token null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1843/runs/` (local, explicit no-sync) | Candidate/control quality and decode work are identical; 2.2% p50 delta is below noise floor. Rejected; never reuse/promote/sync/ship ([results](docs/design/autotrain-cycle-1843-structure-token-null.md)) |
| Autotrain c1842 semantic-exhaustive efficiency-only | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1842/runs/` (local, explicit no-sync) | Candidate cuts tokens 36%, forwards 33%, and p50 39%, but structure falls `.0033` and all semantic/exact metrics are unchanged. Rejected as quality; never reuse/promote/sync/ship ([results](docs/design/autotrain-cycle-1842-semantic-exhaustive-efficiency-only.md)) |
| Autotrain c1840/c1841 tail promotion | 2 size-matched 1,608,962-param scratch checkpoints | c1840 trains + c1841 hash-linked eval replay under `outputs/autoresearch/` (local, explicit no-sync) | Executed replacement-sampling tail comparison is a held-out null; c1844 audit invalidated it as promotion evidence for the c1830 capacity-aware source because the queue dropped that sampling lever. Never promote/sync/ship ([results](docs/design/autotrain-cycle-1841-promotion-held-out-null.md)) |
| Autotrain c1839 capacity-aware tail third-seed null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1839/runs/` (local, explicit no-sync) | Candidate/control are quality- and work-identical; candidate p50 is 3.8% slower. Tail effect is seed-sensitive, so this screening arm is rejected; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1839-capacity-aware-tail-null.md)) |
| Autotrain 8c0b60dd-c2 component-plan structural win, 3rd reproduction (session j48f8u) | 2 size-matched scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2/runs/` (local, explicit no-sync) | `component-plan` beats control on `structural_similarity` by `+.05613`, byte-identical to two prior sessions' independent measurements (PR #1369, `autotrain-cycle-c4-...`). Screening only; fresh-seed confirmation of the same hypothesis remains blocked by two documented harness bugs, not fixed here. Never reusable/promoted/synced/ship ([results](docs/design/continuous-openui-local-j48f8u-c2-results.md)) |
| Autotrain 8c0b60dd-c1 bounds/control exact tie (session j48f8u) | 2 size-matched 1,681,794-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1/runs/` (local, explicit no-sync) | Byte-identical training loss and exact smoke-metric tie between arms; only p50 latency differs (`1790.84` vs `1922.47` ms), which alone is not a metric win. Never reusable/promoted/synced/ship ([results](docs/design/continuous-openui-local-j48f8u-c1-results.md)) |
| Autotrain c1838 capacity-aware tail confirmation | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-202607-98199209-c1838/runs/` (local, explicit no-sync) | Fresh seed repeats structure `.4019→.4372` and binder F1 `.8000→.8222`; fidelity/reward regress slightly and p50 rises 5.1%. Confirmed only for promotion-suite and Lean preflight; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1838-capacity-aware-tail-confirmed.md)) |
| Autotrain c1832 incomplete confirmation | 1 candidate-only 1,608,962-param scratch checkpoint; control stopped at step 2/20 | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1832/runs/` (local, explicit no-sync) | No comparison: candidate smoke 0/3 complete with 3 decode timeouts; control had no checkpoint or eval. Endpoint-boundary harness failure; never reuse/promote/sync/ship ([results](docs/design/autotrain-cycle-1832-confirmation-endpoint-timeout.md)) |
| Autotrain c1830 capacity-aware tail fixture positive | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1830/runs/` (local, explicit no-sync) | Structure `.3506→.4033`, binder F1 `.7222→.8222`, fidelity `.6111→.7222`, tokens `104→99`, forwards `23→22`; fresh-seed confirmation required. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1830-capacity-aware-tail-positive.md)) |
| Autotrain c1829 capacity-aware quality/cost | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1829/runs/` (local, explicit no-sync) | Effective exposure `24.82→32.27`, structure `.4197→.5300`, MPR `.3333→.6667`, recall `.1667→.4167`; p50 regresses `2839→4680` ms. Rejected over latency budget; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1829-capacity-aware-quality-cost.md)) |
| Autotrain c1828 wide-draft null | 2 size-matched 1,608,962-param scratch checkpoints (byte-identical) | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1828/runs/` (local, explicit no-sync) | Quality/work identical; draft 16 p50 `2931` vs `2959` ms is only 0.97% faster and compiler time is worse. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1828-wide-draft-null.md)) |
| Autotrain c1827 compiler-cache null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1827/runs/` (local, explicit no-sync) | Quality/work identical; cached p50 `2851` vs `2870` ms is only 0.67% faster and below the 5% floor, with zero shared-domain hits. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1827-compiler-cache-null.md)) |
| Autotrain c1826 bounded-margin null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1826/runs/` (local, explicit no-sync) | Candidate/control are quality- and work-identical: structure `.3439`, MPR `.6667`, tokens `201`, forwards `51`; candidate p50 `11719` vs `11584` ms. Reject bounds on compiler-tree path; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1826-bounded-margin-null.md)) |
| Autotrain c1824 compiler-decision-margin quality/cost tradeoff | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1824/runs/` (local, explicit no-sync) | Candidate improves structure `.1353→.4811`, MPR `.3333→.6667`, recall `.1667→.4167`, and binder F1 `.6333→.8222`, but tokens `21→61`, forwards `4→15`, and p50 `973→3902` ms. Rejected over latency budget; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1824-compiler-decision-margin-quality-cost.md)) |
| Autotrain c1823 compiler-decision-token fresh-seed null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1823/runs/` (local, explicit no-sync) | Candidate/control are identical on structure `.0575`, MPR `0`, recall `0`, binder F1 `.4889`, fidelity `.3889`, and reward `0`; candidate is 0.92% slower. Reject as non-reproducible; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1823-compiler-decision-token-fresh-seed-null.md)) |
| Autotrain c1822 compiler-decision-token fixture positive | Reuses c1821's 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1822/runs/` (local, explicit no-sync) | Candidate improves structure `.05237→.14623`, MPR `0→.6667`, recall `.0833→.3333`, and p50 `14011.85→12386.03` ms. Fresh-seed confirmation required; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1822-compiler-decision-token-positive.md)) |
| Autotrain c1821 compiler-decision-token incomplete | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1821/runs/` (local, explicit no-sync) | Both 3-document batches timed out, so all quality metrics are unmeasured. Candidate training covered 34 compiler-decision rows; exact frozen replay under eval v78/campaign v119 required. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1821-compiler-decision-token-measurement-incomplete.md)) |
| Autotrain c1819 component-edge-margin reject | 1 new + 1 reused size-matched 1,608,962-param scratch checkpoint | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1819/runs/` (local, explicit no-sync) | Candidate is faster but regresses structure `.40443→.17417`, MPR `.6667→.3333`, and binder F1 `.95238→.63333`. v117 invalidates the original efficiency-positive label. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1819-component-edge-margin-rejected.md)) |
| Autotrain c1818 component-edge-margin incomplete | 1 completed 1,608,962-param control checkpoint; candidate absent | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1818/runs/` (local, explicit no-sync) | Candidate was rejected by an omitted `ModelBuildConfig` allowlist owner before model construction; control-only metrics are not a treatment comparison. Exact frozen replay required; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1818-component-edge-margin-harness-failure.md)) |
| Autotrain c1817 component-edge-token null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1817/runs/` (local, explicit no-sync) | Candidate/control are quality-identical: structure `.17417`, MPR `.3333`, binder F1 `.6333`, AST/canonical 0; 0.25% efficiency delta is noise. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1817-component-edge-token-null.md)) |
| Autotrain c1812 promotion measurement | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1812/runs/` (local, explicit no-sync) | Candidate completes smoke/held-out, but control times out 3/3 and 5/5; matched effect is unavailable. Lean preflight proved; exact frozen replay required. Never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1812-promotion-measurement-incomplete.md)) |
| Autotrain c3 grammar-completion-bounds quality-neutral | 2 size-matched 1,608,962-param scratch checkpoints (frozen replay of c2) | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3/runs/` (local, explicit no-sync) | First cycle in this loop to complete train+eval end to end; bounds vs control tie on smoke structure `.0575` both, bounds `42.91` ms slower p50; rejected, quality-neutral, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-c3-bounds-quality-neutral.md)) |
| Autotrain c2 AgentV/NODE_OPTIONS infra failure | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2/runs/` (local, explicit no-sync) | Both arms trained (loss `22.6219` each) but `--ship-gates` eval crashed on a missing AgentV SDK before any scoreboard; not scoreable, not a model result; never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-c2-agentv-missing-infra-failure.md)) |
| Autotrain c1810 balanced-close confirmation | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1810/runs/` (local, explicit no-sync) | Fresh seed repeats structure `.0964→.17417`, MPR `0→.3333`, binder F1 `0→.6333`, recall `.0833→.25`; p50 `906.83→983.80` ms. Confirmed only for promotion suite, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1810-balanced-container-close-confirmed.md)) |
| Autotrain c1809 balanced-close screen | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1809/runs/` (local, explicit no-sync) | Structure `.0575→.17417`, MPR `0→.3333`, recall `0→.25`, fidelity `0→.5278`; p50 `944.03→973.43` ms. Queued for fresh-seed confirmation, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1809-balanced-container-close-positive.md)) |
| Autotrain c1808 container-close null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1808/runs/` (local, explicit no-sync) | All quality and decode-work metrics tie; candidate training is `3.07→9.35` s and p50 is `1900.30→1931.99` ms. Rejected, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1808-container-close-null.md)) |
| Autotrain c1807 typed-family balance reject | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1807/runs/` (local, explicit no-sync) | Structure `.0575→.10447`, binder F1 `.6333→1.0`, and recall `0→.0833` improve, but MPR stays 0 and p50 rises `1084.71→5868.74` ms; rejected, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1807-typed-family-balance-rejected.md)) |
| Autotrain c1806 STRUCT-token rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1806/runs/` (local, explicit no-sync) | STRUCT weighting lowers family CE but regresses structure `.1725→.1375` and raises p50 `2425.58→7651.84` ms; rejected, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1806-structure-token-rejected.md)) |
| Autotrain c1804/c1805 component-token reject | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1804/runs/` (local, explicit no-sync; c1805 reuses frozen train) | Control timeout 3/3 reproduced; candidate completes but structure `.08173` fails and p50 is `7186.02` ms. Runtime-specific unblock, absolute quality reject; neither checkpoint reusable, promoted, synced, or ship ([replay](docs/design/autotrain-cycle-1805-component-token-rejected.md)) |
| Autotrain c1803 scaffold-prefix null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1803/runs/` (local, explicit no-sync) | Prefix weight 0 vs 1 ties smoke structure `.419733`, binder F1 `.8222`, recall `.1667`, fidelity `.7222`, and reward `.85367`; treatment worsens p50 by 174.21 ms. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1803-scaffold-prefix-null.md)) |
| Autotrain c1802 DESIGN dropout rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1802/runs/` (local, explicit no-sync) | Dropout `.25` regresses smoke structure `.174167→.096400`, recall `.25→.0833`, and meaningful-program rate `.3333→0`; 2.71 ms faster p50 does not override quality. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1802-design-dropout-rejected.md)) |
| Autotrain c1801 symbol-boundary null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1801/runs/` (local, explicit no-sync) | Boundary weight 0 vs 1 ties smoke structure `.135267`, binder F1 `.6333`, fidelity `.5278`, recall `.1667`, and reward `.76533`; treatment worsens loss and p50. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1801-symbol-boundary-null.md)) |
| Autotrain c1800 mixed-mask null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1800/runs/` (local, explicit no-sync) | Random and mixed masking tie smoke structure `.135267`, binder F1 `.6333`, fidelity `.5278`, recall `.1667`, and reward `.76533`; mixed masking is slower and has worse loss. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1800-mixed-mask-null.md)) |
| Autotrain c1799 slot-augmentation rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1799/runs/` (local, explicit no-sync) | Slot augmentation regresses smoke structure `.13750→.05750`, binder F1 `.8222→.6333`, and fidelity `.7222→.5278`; faster p50 does not override quality. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1799-slot-augmentation-rejected.md)) |
| Autotrain c1796 semantic-contrast incomplete | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1796/runs/` (local, explicit no-sync) | Both arms have typed decode timeouts (control smoke/held-out complete 1/3 and 0/5; treatment 1/3 and 4/5), so partial quality is non-attributable. Incomplete; exact frozen replay only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1796-semantic-contrast-incomplete.md)) |
| Autotrain c1795 edge-alignment null | 2 size-matched 1,766,987-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1795/runs/` (local, explicit no-sync) | Exact smoke quality tie at meaning/structure/binder F1/recall/fidelity/reward `0/.05750/.6333/0/.5278/.76533`; treatment p50 improves 3.71%, below the 5% floor, while loss and train wall worsen. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1795-edge-alignment-null.md)) |
| Autotrain c1793 fidelity confirmation rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1793/runs/` (local, explicit no-sync) | Fresh-seed fidelity 1.5 regressed structure `.45750→.44583`, binder F1 `.9524→.6333`, fidelity `.9167→.5278`, and reward `.92000→.80333` despite faster p50. Confirmation rejected; false-positive queue gate repaired; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1793-fidelity-confirmation-rejection.md)) |
| Autotrain c1791 fidelity fixture candidate | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1791/runs/` (local, explicit no-sync) | Fidelity weight 0.5→1.5 improved meaning `0→.3333`, structure `.14593→.17417`, binder F1 `.5→.6333`, recall `0→.25`, fidelity `.4444→.5278`, and p50 `1,154.30→1,088.81` ms. Fixture candidate only; fresh confirmation required; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1791-fidelity-candidate.md)) |
| Autotrain c1789 binder-component-plan incomplete screen | 2 size-matched 1,897,922-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1789/runs/` (local, explicit no-sync) | Control has one typed decode timeout, so candidate structure/MPR gains are non-attributable. Candidate is also 31.1% slower at p50 and binder F1 is lower. Incomplete; exact frozen replay only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1789-binder-component-plan-incomplete.md)) |
| Autotrain c1788 binder-arity null | 2 size-matched 2,145,602-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1788/runs/` (local, explicit no-sync) | Exact smoke quality tie at parse/meaning/structure/binder F1/recall/fidelity/reward `1/0/.20583/0/.0833/0/0`; candidate p50 improves 3.72%, below the 5% screen, with 30.4% worse loss. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1788-binder-arity-null.md)) |
| Autotrain c1786 steps latency rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1786/runs/` (local, explicit no-sync) | Doubling steps raises structure `.13527→.41973` but leaves meaning/binder F1 unchanged and raises p50 154.1%. Rejected; selector and latency-budget harness repaired; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1786-steps-latency-rejection.md)) |
| Autotrain c1785 recycled bounds null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1785/runs/` (local, explicit no-sync) | Exact smoke quality tie; bounds p50 improves only 1.03%, below the 5% screen, and train wall rises 23.5%. Rejected; lineage-exhaustion harness repaired; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1785-bounds-recycle-null.md)) |
| Autotrain c1784 component-plan confirmation rejection | 2 size-matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1784/runs/` (local, explicit no-sync) | Exact smoke/held-out quality ties; held-out p50 improves only 2.92%, below the 5% efficiency screen, and train wall is 2.38x. c1783 rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1784-component-plan-confirmation-rejection.md)) |
| Autotrain c1783 component-plan efficiency candidate | 2 size-matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1783/runs/` (local, explicit no-sync) | Exact smoke quality tie; candidate p50 1,059.20 vs control 1,126.88 ms gives MPR/ms +6.39%, above the 5% fixture screen. Gates fail. Fresh confirmation only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1783-component-plan-efficiency-candidate.md)) |
| Autotrain c1782 runtime-flag harness failure | 2 size-matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1782/runs/` (local, explicit no-sync) | Both arms trained, but eval rejected an unregistered runtime flag before any scoreboard. Incomplete; exact frozen replay only after campaign v85 / flags v2 / eval v76 repair; neither checkpoint promoted, synced, or ship ([results](docs/design/autotrain-cycle-1782-runtime-flag-harness-failure.md)) |
| Autotrain c1781 canvas invalid comparison | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1781/runs/` (local, explicit no-sync) | Control trained with canvas off but evaluated with canvas on; the reported 3.76% slowdown is non-attributable. Harness fixed in campaign v84. Non-scoreable; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1781-canvas-invalid.md)) |
| Autotrain c1780 component-structure null | 2 size-matched 1,913,789-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1780/runs/` (local, explicit no-sync) | Exact smoke/held-out quality ties; candidate p50 is 6.37% faster smoke but 2.38% slower held-out, and train wall is 2.38x. Gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1780-component-structure-null.md)) |
| Autotrain c1779 binder-topology null | 2 size-matched 2,137,346-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1779/runs/` (local, explicit no-sync) | Exact smoke quality tie; binder topology is 3.80% slower at p50 and trains 2.37x as long. Gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1779-binder-topology-null.md)) |
| Autotrain c1778 batch-size-one runtime null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1778/runs/` (local, explicit no-sync) | Exact smoke quality tie; batch 1 is 7.82% slower at p50 and has worse loss. Gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1778-batch1-runtime-null.md)) |
| Autotrain c1777 component-inventory confirmation rejection | 2 size-matched 1,682,363-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1777/runs/` (local, explicit no-sync) | Fresh-seed smoke quality ties exactly at meaning/binder/fidelity/reward 0 and structure .05750; candidate is 6.61% slower. c1776 does not confirm. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1777-component-inventory-confirmation-rejection.md)) |
| Autotrain c1776 component-inventory candidate | 2 size-matched 1,682,363-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1776/runs/` (local, explicit no-sync) | Held-out meaning rises `0→.2` and structure `.06024→.10690`, but binder F1 falls `.6648→.4371` and p50 rises 17.1%; smoke meaning/structure tie. Gates fail. Fresh confirmation only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1776-component-inventory-candidate.md)) |
| Autotrain c1775 component-edge null | 2 size-matched 1,766,987-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1775/runs/` (local, explicit no-sync) | Exact smoke quality tie; component-edge lowers p50 1,093.01→1,052.15 ms, but MPR/ms improves only 3.88%, below the preregistered 5% minimum. Gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1775-component-edge-null.md)) |
| Autotrain c1773 literal-close runtime unblock | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1773/runs/` (local, explicit no-sync) | Tail-supervised candidate completed 3/3 at p50 1,097.20 ms; control timed out 3/3 on an open nested component trajectory. Candidate absolute meaning/binder/fidelity/reward are 0, and control quality is unavailable. Exact frozen replay only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1773-literal-close-runtime-unblock.md)) |
| Autotrain c1772 bounds confirmation rejection | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1772/runs/` (local, explicit no-sync) | Exact smoke/held-out quality ties; bounds is only 2.68% faster smoke and 1.88% slower held-out. The c1771 fixture speed signal does not confirm. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1772-bounds-confirmation-rejection.md)) |
| Autotrain c1771 bounds efficiency candidate | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1771/runs/` (local, explicit no-sync) | Exact smoke quality/prediction tie; bounds improves fixture MPR/ms 6.16% (p50 1,522.25→1,433.96 ms) at n=3. Gates fail. Fresh confirmation only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1771-bounds-efficiency-candidate.md)) |
| Autotrain c1770 component-edge incomplete | 2 size-matched 1,766,987-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1770/runs/` (local, explicit no-sync) | Both arms timed out on 3/3 smoke documents while extending numeric literals, so all quality metrics are unavailable. The matched telemetry does not support a cache regression; run the registered fresh-seed literal-close arm next. Neither checkpoint is reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1770-component-edge-incomplete.md)) |
| Autotrain c1766 component-plan incomplete | 2 size-matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1766/runs/` (local, explicit no-sync) | Candidate timed out on 3/3 smoke documents after plan bias changed three choices and expanded 109,682 completion states; control completed 3/3. Primary quality is unavailable. Exact content-bound frozen replay only; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1766-component-plan-incomplete.md)) |
| Autotrain c1765 numeric literal-margin regression | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1765/runs/` (local, explicit no-sync) | Direct numeric-close alignment activates on 3 rows and clears the observed margin violation, but meaning falls .3333->0, structure falls 22.17%, binder F1 falls .5->0, and training takes 2.45x wall time. Gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1765-literal-margin-regression.md)) |
| Autotrain c1764 literal-close null | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1764/runs/` (local, explicit no-sync) | Tail weight 2.0 changes checkpoint bytes but is prediction- and quality-identical to weight 0 on smoke n=3 and held-out n=5. All records complete, AgentV 0/2, gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1764-literal-close-null.md)) |
| Autotrain c1760 batch-size-one incomplete measurement | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1760/runs/` (local, explicit no-sync) | Candidate eval interrupted and is explicitly non-scoreable; control completes smoke/held-out AgentV but fails gates. Exact frozen replay only; neither checkpoint is reusable beyond that replay, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1760-batch1-incomplete.md)) |
| Autotrain c1759 doubled-steps confirmation | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1759/runs/` (local, explicit no-sync) | Fresh-seed confirmation rejects c1757: 44 steps cuts loss 77.00% but meaning falls .3333→0, structure falls 66.99%, recall falls .25→0, and p50 worsens 134.51%. Gates fail; neither checkpoint is reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1759-steps-confirmation.md)) |
| Autotrain c1757 steps efficiency candidate | 2 size-matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1757/runs/` (local, explicit no-sync) | Doubling steps improves MPR/ms 13.66% and p50 12.02%, but structure drops 22.33% and recall falls. Gates fail. Candidate is queued for fresh confirmation only; not promoted, synced, or ship ([results](docs/design/autotrain-cycle-1757-steps-efficiency-candidate.md)) |
| Autotrain c1756 combined runtime diagnostic | 2 exactly matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1756/runs/` (local, explicit no-sync) | Exact smoke/held-out quality and loss tie; bounds+canvas is 1.90% faster smoke but 7.79% slower held-out. AgentV complete and gates fail. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1756-combined-runtime-diagnostic.md)) |
| Autotrain c1755 compact-canvas diagnostic | 2 exactly matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1755/runs/` (local, explicit no-sync) | Exact smoke quality/loss tie; canvas p50 is 6.08% slower despite 6.63% lower train wall. AgentV complete and gates fail. Rejected runtime diagnostic; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1755-compact-canvas-diagnostic.md)) |
| Autotrain c1754 completion-bounds diagnostic | 2 exactly matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1754/runs/` (local, explicit no-sync) | Exact smoke quality/loss tie; bounds p50 is 0.15% slower and train wall 1.19x. Rejected runtime diagnostic; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1754-completion-bounds-diagnostic.md)) |
| Autotrain c1753 coupled component-structure screen | 2 exactly matched 1,913,789-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1753/runs/` (local, explicit no-sync) | Joint plan+edge coupling reduces smoke structure `.05750→.04333`, leaves meaning 0, raises p50 20.17%, takes 2.68x train wall, and worsens loss. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1753-coupled-component-structure-screen.md)) |
| Autotrain c1752 coupled binder-topology promotion-cadence screen | 2 exactly matched 2,137,346-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1752/runs/` (local, explicit no-sync) | Both 22-step CPU arms are exact smoke n=3 and held-out n=5 quality ties. The coupled treatment is 6.03% slower smoke, 0.89% slower held-out, takes 2.22x training wall, and has worse loss. Gates fail; no champion/Lean promotion claim. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1752-coupled-binder-topology-screen.md)) |
| Autotrain c1751 coupled-topology incomplete screen | 1 control-only 2,137,346-param scratch checkpoint; candidate has none | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1751/runs/` (local, explicit no-sync) | Control trained 24 steps but its eval process exceeded the stage wall; candidate failed pretrain because compiler-path decode lacked its required mode. No scoreable comparison. Harness failure; control checkpoint provenance-only, never reusable/promoted/synced/ship ([results](docs/design/autotrain-cycle-1751-coupled-topology-incomplete.md)) |
| Autotrain c1750 component-inventory screen | 2 exactly matched 1,682,363-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1750/runs/` (local, explicit no-sync) | Both 23-step CPU arms are exact smoke n=3 quality ties. Treatment p50 improves 3.50%, below the 5% floor, and loss is worse. Non-zero auxiliary targets with decode weight 0 trigger a coupled train/decode harness successor. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1750-component-inventory-screen.md)) |
| Autotrain c1749 component-edge screen | 2 exactly matched 1,766,987-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1749/runs/` (local, explicit no-sync) | Both 22-step CPU arms are exact smoke n=3 quality ties: parse 1.0, meaning .3333, structure .17417, recall .25. Treatment p50 improves only 0.79%, below the 5% floor, with worse loss and 1.16× training wall. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1749-component-edge-screen.md)) |
| Autotrain c1748 component-plan promotion-cadence screen | 2 exactly matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1748/runs/` (local, explicit no-sync) | Both 24-step CPU arms are exact smoke n=3 and held-out n=5 quality ties with zero meaning/component recall/AST F1. Treatment is 3.51% faster on smoke but 1.60% slower held-out, takes 2.19× training wall, and has worse loss. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1748-component-plan-screen.md)) |
| Autotrain c1737 compact-canvas incomplete screen | 2 exactly matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1737/runs/` (local, explicit no-sync) | Both 22-step CPU trains completed, but both smoke evals stopped at 2/3 records with non-scoreable progress only. No quality, latency, AgentV, or gate comparison exists. Inconclusive; exact frozen replay only, never promoted/synced/ship ([results](docs/design/autotrain-cycle-1737-canvas-timeout.md)) |
| Autotrain c1736 completion-bounds promotion-cadence screen | 2 exactly matched 1,608,962-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1736/runs/` (local, explicit no-sync) | Both 24-step CPU arms are exact smoke n=3 and held-out n=5 quality ties with zero meaning; bounds is 4.77% / 3.91% slower and takes 1.35× training wall. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1736-bounds-promotion-screen.md)) |
| Autotrain c1735 component-structure screen | 2 exactly matched 1,913,789-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1735/runs/` (local, explicit no-sync) | Both 23-step CPU arms are exact smoke n=3 quality ties; treatment p50 is 5.30% slower (1,509.10→1,589.07 ms), training wall is 3.04×, and loss is worse. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1735-component-structure-screen.md)) |
| Autotrain c1734 component-inventory screen | 2 exactly matched 1,682,363-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1734/runs/` (local, explicit no-sync) | Both 22-step CPU arms are exact smoke n=3 quality ties: parse 1.0, meaning .3333, structure .17417, binder F1 .6333, recall .25. Treatment p50 is only 0.47% faster (1,574.83→1,567.47 ms), below the 5% floor. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1734-component-inventory-screen.md)) |
| Autotrain c1733 component-edge screen | 2 exactly matched 1,766,987-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202607-98199209-c1733/runs/` (local, explicit no-sync) | Both 24-step CPU arms are exact smoke n=3 quality ties: parse/binder F1 1.0, meaning 0, structure .20693, recall .08333. Treatment p50 is 2.13% faster (6,136.43→6,005.56 ms), below the 5% floor. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1733-component-edge-screen.md)) |
| Autotrain c1732 component-plan promotion-cadence screen | 2 exactly matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1732/runs/` (local, explicit no-sync) | Both 23-step CPU arms are exact quality ties on smoke n=3 and held-out n=5. Treatment is 0.46% slower on smoke and 3.96% faster on held-out with zero held-out meaning and 2.89× training wall. Rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1732-component-plan-promotion-screen.md)) |
| Autotrain c1731 component-plan replication | 2 exactly matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1731/runs/` (local, explicit no-sync) | Both 22-step CPU arms have identical parse 1.0, meaningful .3333, structure .4197, binder F1 .6333, and recall .1667. Treatment p50 improved only 0.65% (3,453.06→3,430.55 ms), below policy v4's 5% efficiency floor. Quality-null/rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1731-component-plan-replication.md)) |
| Autotrain c1730 component-plan screen | 2 exactly matched 1,755,764-param scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1730/runs/` (local, explicit no-sync) | Both 24-step CPU arms completed smoke n=3 with identical parse 1.0, meaningful .3333, structure .1742, binder F1 .6333, and recall .25. Treatment p50 was 1,560.93 vs 1,798.92 ms, an efficiency-only fixture signal queued for confirmation. Quality-null; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1730-component-plan-screen.md)) |
| Autotrain c1729 binder-topology screen | 1 control checkpoint (1,608,962 params) + 1 treatment checkpoint (2,137,346 params) | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1729/runs/` (local, explicit no-sync) | Both 23-step CPU scratch arms completed smoke n=3 with parse 1.0, structure .0575, and meaningful/binder F1 0.0. Treatment changed no output or quality metric and was +32.84% trainable params, exposing invalid size-match accounting. Null/rejected; neither checkpoint reusable, promoted, synced, or ship ([results](docs/design/autotrain-cycle-1729-binder-topology-capacity-audit.md)) |
| Autotrain c1717 frozen replay | 1 completed 1,608,962-param CPU scratch control checkpoint; batch1 training incomplete | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1717/runs/` (local, explicit no-sync) | Control smoke n=3 completed (parse/binder F1 1.0, meaningful .3333, structure .3656) and honestly failed gates. Batch1 exhausted its stage redundantly retraining the already-identical frozen checkpoint and never reached eval. Incomplete, not reusable/promoted/ship; stage-level frozen reuse required ([results](docs/design/autotrain-cycle-1717-redundant-retrain-timeout.md)) |
| Autotrain c1716 frozen replay | 2 matched 1,608,962-param CPU scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1716/runs/` (local, explicit no-sync) | Both deterministic 80-step trains reproduced c1715 checkpoint hashes. Control smoke n=3 completed (parse/binder F1 1.0, meaningful .3333, structure .3656) and honestly failed gates; batch1 eval again timed out, now in official AST parsing from terminal-witness search. Incomplete, not reusable/promoted/ship; exact replay required ([results](docs/design/autotrain-cycle-1716-terminal-parse-timeout.md)) |
| Autotrain c1715 frozen replay | 2 matched 1,608,962-param CPU scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c1715/runs/` (local, explicit no-sync) | Both 80-step trains completed. Control smoke n=3 completed (parse/binder F1 1.0, meaningful .3333, structure .3656) and honestly failed gates; batch1 eval timed out in terminal-witness expansion with no metrics. Incomplete, not reusable/promoted/ship; exact replay required ([results](docs/design/autotrain-cycle-1715-terminal-witness-timeout.md)) |
| Autotrain c1708–c1710 infrastructure diagnostics | 6 matched 1,608,962-param CPU scratch checkpoints | `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c170{8,9,10}/runs/` (local, explicit no-sync) | All declared train steps completed, but a bounded stdout-tail parsing defect stopped the harness before eval; AgentV, metrics, and gates are unavailable. Provenance only—never reuse, promote, or ship; frozen c1710 replay required ([results](docs/design/autotrain-cycles-1708-1710-runtime-artifact-recovery.md)) |
| CAP1 cached zero-parameter typed solver | 8 solver manifests; no neural checkpoint | `outputs/experiments/cap1-cached-zero-parameter-solver-successor-20260729/` (local, explicit no-sync) | Score/prediction-identical to the uncached 0-param solver, but immutable frame/schema memoization cuts elapsed 27.802→15.134s (−45.6%; −59.0% vs trained deterministic). Schema arms remain identical and n=32 underpowered; retained runtime baseline, `CERT_CAP1` rejected, not learned/ship ([results](docs/design/cap1-cached-zero-parameter-solver-successor-20260729.md)) |
| CAP1 zero-parameter typed solver | 8 solver manifests; no neural checkpoint | `outputs/experiments/cap1-zero-parameter-solver-successor-20260729/` (local, explicit no-sync) | Corrected v2 suite with 0 params/steps/train rows/forwards. Schema scores exactly match the trained deterministic arm (relevant/invariant/CAP0 1.0, canonical .800) in 27.8s vs 36.9s, but all schema arms remain identical and n=32 underpowered. Smallest sufficient fixture baseline; `CERT_CAP1` rejected, not learned/ship ([results](docs/design/cap1-zero-parameter-solver-successor-20260729.md)) |
| CAP1 deterministic typed candidate successor | 8 matched scratch scorer checkpoints | `outputs/experiments/cap1-deterministic-typed-candidate-successor-20260729/` (local, explicit no-sync) | Corrected v2 suite; schema arms reach 1.0 relevant/invariant/CAP0 and +.3125 vs NL (`p=.00195`) through 24 fact + 4 exact + 4 accepted-set bypasses and 8 guard abstentions, with **zero model forwards**. All schema arms are identical, so trained params are non-load-bearing; `CERT_CAP1` rejected and checkpoints must never be reused/promoted/ship ([results](docs/design/cap1-deterministic-typed-candidate-successor-20260729.md)) |
| CAP1 typed-fact candidate successor | 8 matched scratch scorer checkpoints | `outputs/experiments/cap1-typed-fact-candidate-successor-20260729/` (local, explicit no-sync) | First valid `cap1_two_pack_freeze/v2` run; same 34,913 params/arm. Schema arms reach 1.0 relevant sensitivity/invariance and +.3125 vs NL (`p=.00195`) with 27 exact fact bypasses and 5 forwards, but filtered-multi equals unfiltered, paired n=32 is underpowered, and filtered-single CAP0 is .75. `CERT_CAP1` rejected; never reuse/promote/ship ([results](docs/design/cap1-typed-fact-candidate-successor-20260729.md)) |
| CAP1 counterfactual-supervision candidate successor | 8 matched scratch scorer checkpoints | `outputs/experiments/cap1-counterfactual-candidate-successor-20260729/` (local, explicit no-sync) | **Invalidated v1-suite fixture:** mini-flow prompts asserted the forbidden mode. Historical relevant sensitivity 0 and filtered-single −.250 are provenance only, not comparable to corrected v2. `CERT_CAP1` rejected; never reuse/promote/ship ([results](docs/design/cap1-counterfactual-candidate-successor-20260729.md)) |
| CAP1 schema-guarded candidate successor | 8 matched scratch scorer checkpoints | `outputs/experiments/cap1-schema-guarded-candidate-successor-20260729/` (local, explicit no-sync) | **Invalidated v1-suite fixture:** mini-flow prompts asserted the forbidden mode. Historical .667 sensitivity and −.250 single-arm effect are provenance only, not comparable to corrected v2. `CERT_CAP1` rejected; never reuse/promote/ship ([results](docs/design/cap1-schema-guarded-candidate-successor-20260729.md)) |
| CAP1 pack-generic candidate successor | 8 matched scratch scorer checkpoints | `outputs/experiments/cap1-pack-generic-candidate-successor-20260729/` (local, explicit no-sync) | **Invalidated v1-suite fixture:** mini-flow prompts asserted the forbidden mode. Historical .55–.725 canonical range is provenance only, not comparable to corrected v2. `CERT_CAP1` rejected; not reusable/promoted/ship ([results](docs/design/cap1-pack-generic-candidate-successor-20260729.md)) |
| Deadline-classification callout diagnostic | `parser-fork-smoke-callout/last.pt` | `outputs/autoresearch/…` (local, no-sync) | Same 1,610,114-param deterministic checkpoint SHA; timeout laundering was removed: smoke/held-out/OOD became explicit runtime timeouts with no fallback output or grammar dead end. The safety repair is retained, but callout meaning and overall gates failed, AgentV was 0/5—not reusable/promoted/ship ([results](docs/design/decode-compiler-tree-deadline-classification-fix.md)) |
| Parser-fork button diagnostic | `parser-fork-smoke-button/last.pt` | `outputs/autoresearch/…` (local, no-sync) | Same 1,610,114-param deterministic checkpoint SHA; button compiler was 14,684.025 ms and total 16,276.909 ms with parse 1.0 and zero fallback. The locked smoke endpoint passed, but held-out/RICO fell back, adversarial timed out, AgentV was 0/5, and compound/overall decisions failed—not reusable/promoted/ship ([results](docs/design/decode-compiler-tree-parser-fork-button-measured-results.md)) |
| Parser-fork aggregate diagnostic | `parser-fork-smoke-hero/last.pt` | `outputs/autoresearch/…` (local, no-sync) | Same 1,610,114-param deterministic checkpoint SHA; exact parser forks cut hero compiler time 23.0% to 21,650.370 ms and total to 23,195.864 ms. Locked runtime endpoint passed, but five subsets were only `n=1`, OOD used certified fallback, AgentV 0/5, and overall gates failed—not reusable/promoted/ship ([results](docs/design/decode-compiler-tree-parser-fork-aggregate-measured-results.md)) |
| Incremental-lex aggregate diagnostic | `incremental-lex-smoke-hero/last.pt` | `outputs/autoresearch/…` (local, no-sync) | Same 1,610,114-param deterministic checkpoint SHA; hero compiler fell another 555.999 ms and total cleared the deadline at 29,812.095 ms with zero fallback. Locked diagnostic endpoint passed, but five subsets were only `n=1`, OOD timed out, AgentV 0/5, and overall gates failed—not reusable/promoted/ship ([results](docs/design/decode-compiler-tree-incremental-lex-aggregate-measured-results.md)) |
| Lexer-cache aggregate diagnostic | `lexer-cache-smoke-hero/last.pt` | `outputs/autoresearch/…` (local, no-sync) | 16-step / 1,610,114-param CPU scratch; hero compiler cost fell 1.84% but total remained 30,003.055 ms with certified fallback, five subsets were only `n=1`, AgentV 0/5 — locked hypothesis rejected, not reusable/promoted/ship ([results](docs/design/decode-compiler-tree-lexer-cache-aggregate-measured-results.md)) |
| SLM-313 AbstractPlan local checkpoint | `slm313_local_plan_1k_v2/last.pt` | `outputs/runs/…` (local, no-sync) | 9-step / 1,006-token CPU scratch plan-head + learned connector; complete locked 6,102-row matrix gives zero meaningful-v2/binder-F1 effect vs destructive controls—rejected, not reusable, promoted, or ship ([evidence](docs/design/abstract-plan-functional-evidence.md)) |
| SLM-322 AP-027 Pareto screening checkpoint | `slm322_ap027_scratch_v1/last.pt` | `outputs/runs/…` (local, no-sync) | 8-step connector-free CPU scratch checkpoint, trained only to load the current symbol_only/v2 output contract; 1-seed screening decode at refinement rounds 1-2, rounds 4/8 and connector-dependent arms pending—wiring only, not promoted or ship ([evidence](docs/design/discrete-plan-pareto.md)) |
| E1211 seed-7 topology-dose control | `e1211_v273_e937_batch4_seed7_lr1e4_binder_topology_quarter/last.pt` | `outputs/runs/…` (local, no-sync) | 395 × 4 CPU scratch draws; strict held `n=5` exactly matches E1182/E1200 (parse/meaning .4, strict .2, fidelity .28, structure .2852, recall .3333, reward .3388, 3 timeouts)—neutral/rejected, not promoted or ship ([results](docs/design/iter-e1211-e1214-seed7-topology-slot-component-20260725.md)) |
| SLM-298 local factorial diagnostics | 20 local `d32/d64` scratch checkpoints | `outputs/runs/slm298_local_factorial*/cells/` (local, no-sync) | 520 strict rows; locked `n=1`; all completed constrained cells syntax 1.0 but strict meaningful/binder F1 0.0; four d32 seed-2 cells cap-censored — rejected, not promoted or ship ([results](docs/design/iter-slm298-capacity-context-curriculum-20260725.md)) |
| SLM-287 five-seed locked baseline | `slm287-trained-local-v13-20260725` (10 cells) | `outputs/runs/…` (local, no-sync) | CPU float32 Choice TwoTower; 97-record strict train snapshot, 5k-token budget, five seeds × scratch design off/on, 226-record locked holdout and AgentV per shard. Meaning-v2/binder F1 stay 0; absolute-probability MDE 2.0 pp. Diagnostic only, not promoted or ship |
| SLM-230 bounded recurrence diagnostic | `slm230_bounded_recursive_r4_r2/last.pt` | `outputs/runs/…` (local) | R=4 scratch checkpoint: SLM-230 is `stagnant`; SLM-231 is `expansive_unstable`; SLM-232 finds z0 rank 2.1054 but rank 0 after context/position removal and vacuous bounded ablations (`unstable`). Rejected, not ship |
| E735 full-head root-arity diagnostic | `e735-symbol-only-root-arity-fullhead140-r1/last.pt` | `outputs/runs/…` (local) | Removes impossible class-41 tail prediction, but weight 0/1 smoke quality remains identical and strict-v2 0.0 — fix retained, checkpoint rejected |
| E733 invalid lexer root-identity attempt | `e733-symbol-only-root-identity140-r1/last.pt` | `outputs/runs/…` (local) | Proposed lever has zero reachable decode applications; config now rejects lexer identity before artifacts — checkpoint invalidated |
| E731 lexer root-arity diagnostic | `e731-symbol-only-root-arity140-r1/last.pt` | `outputs/runs/…` (local) | Lexer-native head is executable, but weights 0/1/2 change no choices; smoke strict-v2 0.0 — checkpoint rejected |
| E714 symbol-only baseline | `e714-symbol-only-scratch600-r1/last.pt` | `outputs/runs/…` (local) | First v2-compatible CPU scratch checkpoint; 600 steps / 48.72s, strict meaning 0.0 and AgentV 0/5 — diagnostic only, not ship |
| E720 component-inventory diagnostic | `e720-symbol-only-component-inventory600-r1/last.pt` | `outputs/runs/…` (local) | Inventory head learned (top-k recall 0.6875), but smoke parse/strict meaning remained 0.0 and weight-4 decode timed out 3/3 — rejected, not ship |
| E721 role/count plan diagnostic | `e721-symbol-only-component-plan190-r4/last.pt` | `outputs/runs/…` (local) | Smoke parse 1.0, but strict meaning 0.0 and plan weight 1 is identical to weight 0; local 190-step syntax diagnostic only, rejected |
| E722 component-edge diagnostic | `e722-symbol-only-component-edge150-r1/last.pt` | `outputs/runs/…` (local) | Parse 1.0 / structure 0.2861 / recall 0.5, but strict meaning 0.0 and edge on/off identical — rejected, not ship |
| E723 slot-owner diagnostic | `e723-symbol-only-slot-owner140-r1/last.pt` | `outputs/runs/…` (local) | Causal smoke + held-out gains; smoke meaning-v1 0.6667 / structure 0.5614, but strict-v2 0.0 — lever retained, checkpoint rejected |
| E725 cumulative inventory diagnostic | `e725-symbol-only-component-inventory130-r1/last.pt` | `outputs/runs/…` (local) | Inventory head learned, but weight 1/0 decode is identical and smoke meaning-v1/strict-v2 0.0 — rejected, not ship |
| E726 invalid root-arity attempt | `e726-symbol-only-root-arity140-r1/last.pt` | `outputs/runs/…` (local) | Choice-only arity lever was unavailable on lexer; tensors match E723 exactly — invalidated, never evaluate/sync/serve |
| E727 binder-arity diagnostic | `e727-symbol-only-binder-arity140-r1/last.pt` | `outputs/runs/…` (local) | Arity head learned, but weights 1/2 change no smoke or held-out choices and strict-v2 remains 0.0 — rejected |
| E729 binder-topology diagnostic | `e729-symbol-only-binder-topology140-r1/last.pt` | `outputs/runs/…` (local) | Topology weights 0.25/1 regress smoke meaning 0.6667→0.3333 and structure 0.5614→0.4642 — rejected |
| Playground demo | `playground_demo/last.pt` | `src/slm_training/resources/checkpoints/playground_demo/` (git) | E497 clean-revision honest smoke: parse/meaningful/fidelity 0.0, structure 0.2203, AgentV 0/5; wiring only |
| Restructure CPU verify | `restructure_cpu_scratch_v0/last.pt` | `outputs/runs/…` (local) | Fixture scratch train OK; smoke parse 0.0 — not ship |
| Local DirectML verify | `local_directml_adreno_20260714/last.pt` | `outputs/runs/…` (local) | Adreno GPU train/checkpoint OK; 5-step wiring run, not evaluated or ship |
| Overnight retrain | `overnight_retrain_200/last.pt` | `/tmp/slm-training-overnight/outputs/runs/…` (local) | 200-step CPU scratch; honest parse 0.0, not ship |
| Overnight retrain extended | `overnight_retrain_1000/last.pt` | `/tmp/slm-training-overnight/outputs/runs/…` (local) | 1,000-step CPU scratch; smoke parse 0.0, not ship |
| E120 singleton diagnostic | `e120_unsandboxed/last.pt` | `outputs/runs/iter-e120-unsandboxed-20260715/…` (local) | 8-step CPU scratch; guarded singleton decode verified, `rico_held n=1` parse 0.0 — not ship |
| E121 judged-corpus E53 iteration | `qx_e53_honest_v5_champion/last.pt` | `outputs/runs/iter-e121d-e53-judged-20260715/…` (local) | 405 judge-approved records; bounded smoke parse 0.0 with decode timeout — not ship |
| E123 judged-corpus 32-step iteration | `e123_judged_32step_b/last.pt` | `outputs/runs/iter-e123b-judged-20260715/…` (local) | 405 judge-approved records; loss 10.97 but smoke parse 0.0 with fallback/canvas cap — not ship |
| E127 schema/slot-contract iteration | `e127_judged_schema_slots/last.pt` | `outputs/runs/iter-e127-schema-slots-20260715/…` (local) | 405 judged records; placeholder validity 0.55 / normalized fidelity 0.25, but parse 0.0 — not ship |
| E128 schema/slot 64-step iteration | `e128_judged_schema_slots_64/last.pt` | `outputs/runs/iter-e128-schema-slots-20260715/…` (local) | Higher LTR/fidelity weights regressed placeholder signals and parse remained 0.0 — not ship |
| E129 schema/slot 64-step low-weight control | `e129_judged_schema_slots_64_lowweights/last.pt` | `outputs/runs/iter-e129-schema-slots-20260715/…` (local) | Lower-weight control also had placeholder/parse 0.0; longer training not justified — not ship |
| E130 schema/slot seed-1 control | `e130_judged_schema_slots_seed1/last.pt` | `outputs/runs/iter-e130-schema-slots-20260715/…` (local) | Seed-1 control had parse and placeholder signals 0.0; E127 not reproducible — not ship |
| E132 generation-focused mixture | `e132_generation_focus/last.pt` | `outputs/runs/iter-e132-generation-focus-20260715/…` (local) | Three-prompt smoke parse/placeholder 0.0; task reweighting rejected — not ship |
| E133 no-fused-LTR path | `e133_no_fuse_ltr/last.pt` | `outputs/runs/iter-e133-no-fuse-ltr-20260715/…` (local) | Three-prompt smoke parse/structure 0.0 with one timeout; fused LTR retained — not ship |
| E135 HF context control | `e135_hf_context_control/last.pt` | `outputs/runs/iter-e135-hf-context-20260715/…` (local) | HF context improves structural/placeholder signals but parse 0.0 with one timeout — not ship |
| E136 HF context 32-step control | `e136_hf_context_32/last.pt` | `outputs/runs/iter-e136-hf-context-20260715/…` (local) | Longer HF run regressed structure/placeholder to 0.0; checkpoint selection next — not ship |
| E137 HF context 16-step midpoint | `e137_hf_context_16/last.pt` | `outputs/runs/iter-e137-hf-context-20260715/…` (local) | Placeholder validity 0.40 and structure 0.2142, parse 0.0; non-monotonic checkpoint trajectory — not ship |
| E138 HF context seed-1 8-step control | `e138_hf_context_seed1_8/last.pt` | `outputs/runs/iter-e138-hf-seed1-20260715/…` (local) | Same recipe as E135 but seed 1: placeholder validity 0.0 and structure 0.1683, parse 0.0 — not ship |
| E139 HF context seed-2 8-step control | `e139_hf_context_seed2_8/last.pt` | `outputs/runs/iter-e139-hf-seed2-20260715/…` (local) | Same recipe as E135 but seed 2: placeholder validity/structure/parse 0.0 with two timeouts — not ship |
| E173 schema-context 32-step control | `e173-schema-context-32step/last.pt` | `outputs/runs/e173-schema-context-32step/…` (local) | Schema/slot context enabled; bounded syntax probe 1.0 but meaningful parse 0.0 — not ship |
| E174 unfrozen-context 8-step control | `e174-unfrozen-context-8step/last.pt` | `outputs/runs/e174-unfrozen-context-8step/…` (local) | Unfrozen context regressed bounded syntax to 0.0; rejected control — not ship |
| E175 retrieval 8-step control | `e175-retrieval-8step/last.pt` | `outputs/runs/e175-retrieval-8step/…` (local) | Retrieval k=4 regressed bounded syntax/parse to 0.0; rejected control — not ship |
| E176 broad-corpus 8-step control | `e176-broad-corpus-8step/last.pt` | `outputs/runs/e176-broad-corpus-8step/…` (local) | 1,417-record corpus regressed bounded syntax/parse to 0.0; rejected control — not ship |
| E177 semantic-judge 32-step control | `e177-semantic-judge-32step/last.pt` | `outputs/runs/e177-semantic-judge-32step/…` (local) | 496 published judge-gated records; E180 bounded decode reaches syntax 1.0 but meaningful parse 0.0 — not ship |
| E181/E184/E191 compiler-alignment diagnostics | `e181-semantic-balanced-32step`, `e184-compiler-aligned-32step`, `e191-full-compiler-aligned-32step` | `outputs/runs/…` (local) | Balanced mixture did not improve quality; component alignment recovered the root, all-branch alignment regressed it; no meaningful parse or promotion — not ship |
| E195/E196 stratified-alignment diagnostics | `e195-stratified-compiler-aligned-32step`, `e196-stratified-compiler-aligned-matched-32step` | `outputs/runs/…` (local) | E195 invalid (mixture unset); matched E196 reaches syntax 1.0 after parser-state fixes but meaningful parse 0.0 — not ship |
| E201 generated-role diagnostic | `e201-role-stratified-compiler-aligned-32step` | `outputs/runs/…` (local) | Grammar/schema role constraints improve component and placeholder signals, but recursive children hit the token cap with parse 0.0 — not ship |
| E205 Lark-terminal diagnostic | `e205-lark-terminal-stratified-32step` | `outputs/runs/…` (local) | Terminal-derived alignment and schema enum paths restore syntax 1.0 without fallback, but empty bound stacks leave meaningful parse 0.0 — not ship |
| E208/E210/E212 contextual-decision diagnostics | `e208-list-occupancy-stratified-32step`, `e210-list-scope-occupancy-stratified-32step`, `e212-contextual-decision-stratified-32step` | `outputs/runs/…` (local) | Contextual root-child supervision recovers a populated root and fidelity signal, but required schema semantics still fail and meaningful parse remains 0.0 — not ship |
| E214/E215 overfiltered schema-judge diagnostic | `e215-schema-role-judged-32step` | `outputs/runs/e215-schema-role-judged-32step/…` (local) | E214 falsely rejected 27 legal optional-null records; E216 syntax 1.0 but meaningful parse 0.0; superseded by E218 — not ship |
| E218/E219 corrected schema-admission diagnostic | `e219-schema-normalized-32step` | `outputs/runs/e219-schema-normalized-32step/…` (local) | Restores 33 valid records and fixes future producers; E220 syntax 1.0, component recall 0.25, meaningful parse 0.0 — not ship |
| E221 task-balanced exposure diagnostic | `e221-canonical-task-balanced` | `outputs/autoresearch/e221-task-balanced-exposure-v4/runs/…` (local) | 32 CPU steps on canonical E218; effective exposure 29.68/128; strict eval failed 9 gates, AgentV 1/5 — not ship |
| E222 capacity-aware exposure diagnostic | `e222-capacity-aware-matched` | `outputs/autoresearch/e222-capacity-aware-exposure/runs/…` (local) | Effective exposure rose to 83.59/128, but strict smoke parse regressed to 0.0 and 10 gates failed — not ship |
| E223 quota-capacity exposure diagnostic | `e223-quota-capacity-matched` | `outputs/autoresearch/e223-quota-capacity-exposure/runs/…` (local) | Task quotas and syntax are deterministic, but semantic metrics are 0.0 and 12 gates failed — not ship |
| E224–E226 semantic alignment + honest tree eval | `e224-semantic-exhaustive-matched` | `outputs/autoresearch/e224-semantic-exhaustive-alignment/runs/…` (local) | Deterministic tree reaches syntax 1.0 on all suites with honest fidelity, but meaningful-program quality fails 5 gates — not ship |
| E227 legal-candidate alignment | `e227-candidate-set-matched` | `outputs/autoresearch/e227-candidate-set-alignment/runs/…` (local) | Candidate loss optimizes, but empty-layout collapse fails 12 gates and AgentV 0/5 — rejected, not ship |
| E228 legal-candidate margin | `e228-candidate-margin-matched` | `outputs/autoresearch/e228-candidate-margin-alignment/runs/…` (local) | Best diagnostic: syntax/contract 1.0, failures reduced to 4, but AgentV 1/5 — not ship |
| E229 64-step margin continuation | `e229-margin-64step` | `outputs/autoresearch/e229-margin-continuation/runs/…` (local) | Syntax restored to 1.0 after generalized literal-frame fix, but the same 4 gates fail — duration rejected, not ship |
| E230 diverse judged roots | `e230-diverse-roots-32step` | `outputs/autoresearch/e230-diverse-judged-roots/runs/…` (local) | Published 126 judge-passed generation roots and verified RICO/human exposure; same 4 gates fail and adversarial regresses — data fix retained, checkpoint rejected, not ship |
| E231 component inventory | `e231-component-inventory-32step` | `outputs/autoresearch/e231-component-inventory/runs/…` (local) | Inventory target learns, but bias-off metrics/component choices are identical; 6 thresholds fail, AgentV 1/5 — rejected, not ship |
| E232 role component plan | `e232-role-component-plan-32step` | `outputs/autoresearch/e232-role-component-plan/runs/…` (local) | Root/count targets learn and improve one adversarial case, but 4 frontier thresholds still fail; stronger calibration has no aggregate gain — rejected, not ship |
| E233 resolved-AST component edges | `e233-component-edges-32step` | `outputs/autoresearch/e233-component-edges/runs/…` (local) | Edge target learns, but edge on/off suite aggregates are identical and 4 thresholds fail — rejected, not ship |
| E234 edge decision alignment | `e234-edge-decision-alignment-32step` | `outputs/autoresearch/e234-edge-decision-alignment/runs/…` (local) | Legal-decision accuracy learns and changes 5 choices, but on/off aggregates are identical and 4 thresholds fail — rejected, not ship |
| E235 binder-instance plan | `e235-binder-instance-plan-32step` | `outputs/autoresearch/e235-binder-instance-plan/runs/…` (local) | Full binder supervision changes 4 legal choices, but on/off aggregates are identical and 9 thresholds fail — rejected, not ship |
| E236 binder topology | `e236-binder-topology-32step` | `outputs/autoresearch/e236-binder-topology/runs/…` (local) | Topology objective fails to learn, changes 0/38 applied choices, and collapses semantic metrics; 12 thresholds fail — rejected, not ship |
| E237 detached topology | `e237-detached-topology-32step` | `outputs/autoresearch/e237-detached-topology/runs/…` (local) | Detaching already-frozen context is a no-op and exactly reproduces E236; 12 thresholds fail — rejected, not ship |
| E238 binder arity (invalidated) | `e238-binder-arity-32step` | `outputs/autoresearch/e238-binder-arity/runs/…` (local) | Optional-head RNG shifted matched training draws; ten thresholds fail and the run is confounded — not ship |
| E239 isolated binder arity | `e239d-binder-arity-fully-isolated-32step` | `outputs/autoresearch/e239-binder-arity-corrected/runs/…` (local) | 104/104 shared tensors match the control; 29 changed choices do not produce meaningful programs; 11 thresholds fail — rejected, not ship |
| E249 exact-event CE plus margin | `qx_e249_local_ce_margin` | `outputs/autoresearch/e249-local-ce-margin/runs/…` (local) | Held-out lexical wins improve sharply, but structure/reward regress on every suite and AgentV is 0/5 — rejected, not ship |
| E252 verifier-backed set FTPO | `qx_e252_local_ftpo_set` | `outputs/autoresearch/e252-ftpo-set/runs/…` (local) | Syntax remains 1.0, but fidelity collapses to 0, structure/reward regress everywhere, and AgentV is 0/5 — rejected, not ship |
| E263 broad gold-AST set FTPO | `qx_e262_broad_gold_ast_ftpo_set` | `outputs/autoresearch/e262-broad-gold-ast-ftpo/runs/…` (local) | Emitted as E262 before ID reconciliation; syntax/fidelity match E248, but held-out loss worsens, structure regresses everywhere, and AgentV is 0/5 — rejected, not ship |
| E264 guarded gold-AST set FTPO | `qx_e264_guarded_gold_ast_ftpo_set` | `outputs/autoresearch/e264-guarded-gold-ast-ftpo/runs/…` (local) | No trained step passed the held-out Pareto guard; restored checkpoint is bit-identical to E228 and current parent control reproduces all metrics — no model gain, not ship |
| E265 safe gold-AST set FTPO | `qx_e265_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e265-safe-gold-ast-ftpo/runs/…` (local) | 3/30 backtracked proposals improve aggregate exact-state metrics, but per-kind regressions are masked and semantic quality falls on most suites — rejected, not ship |
| E266 stratified safe set FTPO | `qx_e266_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e266-stratified-safe-gold-ast-ftpo/runs/…` (local) | Per-decision-kind guard rejects all 30 global FTPO proposals; parent is restored exactly, while batched validation is 37.7× faster — no model gain, not ship |
| E267 block-coordinate safe set FTPO | `qx_e267_block_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e267-block-stratified-safe-ftpo/runs/…` (local) | Averaging gradients within each decision kind still yields 0/30 safe proposals; parent is restored exactly — no model gain, not ship |
| E268 projected safe set FTPO | `qx_e268_projected_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e268-projected-stratified-safe-ftpo/runs/…` (local) | PCGrad projects 2,220 conflicting task pairs but still yields 0/30 safe proposals; parent restored exactly, 38m59s CPU stage — rejected, not ship |
| E269 MGDA safe set FTPO | `qx_e269_mgda_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e269-mgda-one-step-final/runs/…` (local) | One-step MGDA certifies common train descent, but all five scales regress held-out decision kinds; full 30-step run rejected, parent restored — not ship |
| E272 MGDA plus SGD preflight | `qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set` | `outputs/autoresearch/e272-mgda-sgd-one-step/runs/…` (local) | Collinear SGD improves aggregate held-out loss, but all scales regress per-kind probability/margin guards; parent restored, no full run — not ship |
| Matrix honest champion | V6 E53 family | `outputs/runs/` + matrix docs | Scratch + limited `rico_held` — not production HF ship |
| P13 matched E50 controls | fixture + integrated E50 | `/tmp/slm17-e50-*-honest/` (local scratch) | Integrated fidelity +0.04 held / +0.0333 RICO; parse 0.0, not ship |
| Frozen X2 baseline | `gx_x2_codec` seeds 0/1/2 | `/tmp/slm-training-fixed-baseline/outputs/topology_baseline/` | Fixed-canvas comparison scored zero on all suites; not ship |
| Topology v2 smoke | `grammar_diffusion_overfit` | pytest temporary checkpoint | n=2 parse/fidelity 0.5, topology composite 0.482; wiring only, not ship |
| Topology X9/X14 confirmation | 6 seed checkpoints | `/tmp/slm-training-grammar-topology/outputs/topology_confirm_4bf964d/` | 200-step CPU scratch; all fail multi-suite gates, no promotion/sync |
| ScopeDiff X18/X21 confirmation | 6 seed checkpoints | `outputs/runs/gx_x{18,21}_*_confirm_200/` (local) | 200-step CPU scratch; all-suite median parse/fidelity 0.0, all fail gates, no promotion/sync |
| EFS0-04 X22 reproduction | `gx_x22_kapur_tree_edit_s0/last.pt` | `outputs/runs/gx_x22_kapur_tree_edit_s0/…` (local) | 80-step seed-0 audit-material replay; SHA `a9cfb450…02ff6`; syntax 1.0 but meaningful parse 0.333/0.2/0/0/0.667 on bounded suites; gates fail, no sync/promotion ([results](docs/design/iter-efs0-04-x22-reproduction-20260717.md)) |
| B3 five-minute lexer control | `capacity_lexer_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/b3-matched-5m-e287-r2/…` (local) | 53-step / 5,004-token CPU scratch; five-suite parse/meaningful 0.0, AgentV 0/5 — not promoted or ship |
| B3 five-minute choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/b3-matched-5m-e287-r2/…` (local) | E288 frozen eval: deterministic parse 1.0 on all suites, but meaningful/fidelity 0.0 and AgentV 0/5 — not promoted or ship |
| E289 cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e289-choice-state-cache/…` (local) | Same checkpoint SHA as E288; exact symbolic-state cache preserves parse 1.0 and cuts p50 2.65×–5.86×, but semantic metrics and AgentV remain zero — not promoted or ship |
| E290 direct-candidate choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e290-choice-direct-candidates/…` (local) | Same checkpoint SHA; exact grammar-derived candidates improve p95 1.14×–1.19× but regress p50, while semantic metrics and AgentV remain zero — not promoted or ship |
| E291 completion-cached choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e291-choice-completion-cache/…` (local) | Same checkpoint SHA; exact completion caching improves p50 1.29×–1.99× and p95 1.51×–1.93× vs E290, but semantic metrics and AgentV remain zero — not model-promoted or ship |
| E292 complete-loss choice arm | `capacity_choice_v1__d64_h2_c1_dn2_t5000_x1__s0/last.pt` | `outputs/ladders/e292-choice-loss-suite-complete-r2/…` (local) | Same checkpoint SHA; all five frozen loss categories now complete (weighted NLL 7.2265), but honest meaningful rate is 0.0 and AgentV is 0/5 — not promoted or ship |
| E293 choice-native component plan | `e293-choice-component-plan-r3/last.pt` | `outputs/runs/e293-choice-component-plan-r3/…` (local) | Plan target learns and legal bias reduces failures 17→13, but matched no-DESIGN meaningful rate is 0.0 and AgentV 0/5 — not promoted or ship |
| E294 no-DESIGN choice control | `e294-choice-no-design-control-r1/last.pt` | `outputs/runs/e294-choice-no-design-control-r1/…` (local) | No-plan control exactly matches E293 bias-off; meaningful 0.0, AgentV 0/5, 17 failures — not promoted or ship |
| E295 DESIGN-dropout choice arm | `e295-choice-design-dropout-r1/last.pt` | `outputs/runs/e295-choice-design-dropout-r1/…` (local) | 50% deterministic context dropout yields adversarial meaningful 0.25 and AgentV 1/5, but four suites remain 0.0 and 14 gates fail — not promoted or ship |
| E396 durable diagnostic checkpoint | `e396-balanced-type-head-continuation-r1/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/` | Exact SHA `feefa056…c2f2eee0`; bucket verified. E498 restores current-main loading and learned-head application (smoke structure 0.27057), but semantic gates and AgentV remain red. Diagnostic, not champion or ship |
| E499 bounded strict-corpus checkpoints | `e499-*-r4/r6/last.pt` | `outputs/runs/e499-…/` (local) | Matched strict-r4 and document-only r6 both regress smoke structure 0.1542→0.0375 and recall 0.25→0.0; AgentV 0/1, no sync or promotion. All seven checkpoint SHAs are in the full card |
| E500 documentized-expression checkpoints | `e500-*-r1/r2/r3-5k/r4-5k/last.pt` | `outputs/runs/e500-…/` (local) | The 260-row projected corpus is clean and diverse, but both matched 1k/5k pairs have structure 0.0375, semantic metrics zero, and AgentV 0/1. Four exact SHAs are in the full card; no sync or promotion |
| E501 E396→E500 warm-start checkpoints | `e501-e396-e500-*/last.pt` | `outputs/runs/e501-…/` (local) | Explicit new-corpus initialization works, but 5k arms forget parent structure; the 1k arm reaches structure 0.2317 with semantic metrics still zero. Three exact SHAs are in the full card; no sync or promotion |
| E502 prior-retention checkpoints | `e502-e396-e500-*/last.pt` | `outputs/runs/e502-…/` (local) | Preserving checkpoint serving priors raises 1k structure to 0.3169 with recall 0.0833, but 5k collapses and all semantic gates remain zero. Four exact SHAs are in the full card; no sync or promotion |
| E503 initialized-weight retention checkpoints | `e503-e396-e500-retention*-5k/last.pt` | `outputs/runs/e503-…/` (local) | Retention cuts RMS drift up to 74% and restores structure to 0.2029, but recall falls to zero and semantic gates remain red. Four exact SHAs are in the full card; no sync or promotion |
| E504 parent-replay checkpoints | `e504-e396-e500-replay*-5k/last.pt` | `outputs/runs/e504-…/` (local) | 50% exact E357 replay raises structure to 0.2469 and cuts drift 10.46%, but semantic gates remain zero; replay plus retention regresses structure. Five exact SHAs are in the full card; no checkpoint sync or promotion |
| E505 replay-loss attribution checkpoint | `e505-e396-e500-replay050-loss-attribution-r1-5k/last.pt` | `outputs/runs/e505-…/` (local) | E511 component-plan weight 4 reaches aggregate meaningful 0.3846 and fidelity 0.6718 across 13 records. E512 rejects slot weight 8; strict semantic and AgentV gates remain red, with no promotion |
| E513 durable slot-role checkpoint | `e513-e396-e500-replay050-slotrole4-focal2-r3-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k/` | Bucket-verified SHA `59253c67…a88a9548`; 5,000 target tokens in 79.6s under the three-minute cap. Matched OOD meaningful 0.0, fidelity 0.4917, structure 0.2750, AgentV 0/1; durable diagnostic, rejected for promotion |
| E515 focal-zero slot-role checkpoint | `e515-e396-e500-replay050-slotrole4-focal0-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e515-e396-e500-replay050-slotrole4-focal0-r1-5k/` | Bucket-verified SHA `97f2e426…24721c1b`; 5,000 target tokens in 105.8s under the three-minute cap. Matched OOD meaningful 0.25, fidelity 0.6583, structure 0.3213, AgentV 0/1; focal 2 rejected, checkpoint not promoted |
| E517 slot-loss-1 context checkpoint | `e517-e396-e500-replay050-slotrole1-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k/` | Bucket-verified SHA `2b572a04…e24b60e3`; 5,000 target tokens in 130.7s under the three-minute cap. Matched OOD meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1; rejected |
| E519 honest slot-context checkpoint | `e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e519-e396-e500-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `d82155b0…6c91805f`; 5,000 target tokens in 103.2s from clean harness v7. Exact E517 quality parity (meaningful 0.0, fidelity 0.4083, structure 0.2250, AgentV 0/1); honest path retained, checkpoint rejected |
| E522 visible-inventory checkpoint | `e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e522-e396-e521-replay050-slotrole1-honest-context-r2-5k/` | Bucket-verified SHA `97cb10f4…bf420ce`; 5,059 target tokens in 120.7s. E523 fidelity 0.8667 and recall 0.2708 improve, but meaningful stays 0.0, structure falls to 0.1955, and AgentV is 0/1; rejected |
| E525 visible-component checkpoint | `e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e525-e396-e524-replay050-slotrole1-honest-context-r2-5k/` | Bucket-verified SHA `dbd11811…e55e4b9`; 5,059 target tokens in 76.7s. E526 recall rises to 0.4167, but fidelity falls to 0.4667, structure to 0.1452, meaningful stays 0.0, and AgentV is 0/1; rejected |
| E528 visible-component-types checkpoint | `e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e528-e396-e527-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `6a2180d7…306976d5`; 5,059 target tokens in 146.8s. E529 meaningful recovers to 0.25 and reward to 0.5778, but structure falls to 0.1136, strict meaning remains 0.0, and AgentV is 0/1; rejected |
| E616 object-frame slot-bias replay (80-step) | `e616-object-property-slot-bias-scratch80-20260720/last.pt` | `outputs/runs/e616-object-property-slot-bias-scratch80-20260720/` (local) | Fresh 80-step CPU scratch loop on E530, loss 26.5243; matched OOD `n=4` eval now parses 4/4 but stays byte-identical because Gallery's array closes empty before any item opens — not ship |
| E620 required-slot coverage replay (800-step) | `e620-required-slot-coverage-scratch800-20260720/last.pt` | `outputs/runs/e620-required-slot-coverage-scratch800-20260720/` (local) | 800 CPU scratch steps in 80.96s, loss 4.0680; OOD treatment fidelity 0.5500, structure 0.4886, strict-v2 0.0, AgentV 0/1. Lower loss regressed E619 generalization — rejected, not ship |
| E548 fresh TwoTower loop | `e548_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e548_training_loop_twotower_scratch_20260720/` (local) | Fresh 8-step CPU scratch loop on E530, loss 39.4267; no eval/sync, wiring only — not ship |
| E547 fresh TwoTower loop | `e547_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e547_training_loop_twotower_scratch_20260720/` (local) | Fresh 7-step CPU scratch loop on E530, loss 35.7431; no eval/sync, wiring only — not ship |
| E546 fresh TwoTower loop | `e546_training_loop_twotower_scratch_20260720/last.pt` | `outputs/runs/e546_training_loop_twotower_scratch_20260720/` (local) | Fresh 6-step CPU scratch loop on E530, loss 40.3390; no eval/sync, wiring only — not ship |
| E545 fresh TwoTower loop | `e545_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e545_training_loop_twotower_scratch_20260719/` (local) | Fresh 5-step CPU scratch loop on E530, loss 42.1226; no eval/sync, wiring only — not ship |
| E544 fresh TwoTower loop | `e544_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e544_training_loop_twotower_scratch_20260719/` (local) | Fresh 4-step CPU scratch loop on E530 after missing prior local checkpoint, loss 42.3848; no eval/sync, wiring only — not ship |
| E543 resumed TwoTower loop | `e543_training_loop_twotower_resume_scratch_20260719/last.pt` | `outputs/runs/e543_training_loop_twotower_resume_scratch_20260719/` (local) | Resumed E542 full-state to step 3 on E530, loss 39.7476; no eval/sync, wiring only — not ship |
| E542 resumed TwoTower loop | `e542_training_loop_twotower_resume_scratch_20260719/last.pt` | `outputs/runs/e542_training_loop_twotower_resume_scratch_20260719/` (local) | Resumed E541 full-state to step 2 on E530, loss 43.6742; no eval/sync, wiring only — not ship |
| E541 TwoTower training-loop iteration | `e541_training_loop_twotower_scratch_20260719/last.pt` | `outputs/runs/e541_training_loop_twotower_scratch_20260719/` (local) | One-step CPU scratch TwoTower loop on E530, loss 36.9158; no eval/sync, wiring only — not ship |
| E540 training-loop sentinel | `e540_training_loop_scratch_20260719/last.pt` | `outputs/runs/e540_training_loop_scratch_20260719/` (local) | One-step CPU scratch stub loop check on E530, loss 0.5; no eval/sync, wiring only — not ship |
| E531 visible-semantic-role checkpoint | `e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/last.pt` | `hf://buckets/TKendrick/OpenUI/checkpoints/e531-e396-e530-replay050-slotrole1-honest-context-r1-5k/` | Bucket-verified SHA `6b8c1abc…74a6154`; 5,059 target tokens in 99.72s. E532 structure improves slightly to 0.1431, but meaningful falls to 0.0, fidelity to 0.4667, reward to 0.3685, strict meaning stays 0.0, and AgentV is 0/1; rejected |
| E542 learned root-arity checkpoint | `e542-e531-root-reference-arity1-r1-24s/last.pt` | `outputs/runs/e542-e531-root-reference-arity1-r1-24s/` (local) | 24-step scratch continuation in 52.93s, SHA `2d5cd4b3…6854c5d8`; OOD `n=4` meaningful 0.50 / fidelity 0.5917 / structure 0.3019, but learned weight 1 is quality-neutral, strict meaning 0.0, AgentV 0/1; no sync or promotion |
| E543 bounded root-arity checkpoint | `e543-e531-root-reference-bounded-r1-24s/last.pt` | `outputs/runs/e543-e531-root-reference-bounded-r1-24s/` (local) | 24-step scratch continuation in 37.17s, SHA `c6be3791…51d7f90`; bounded loss improves calibration, but OOD `n=4` decisions and quality exactly match E542, strict meaning 0.0, AgentV 0/1; no sync or promotion |
| E544 root-identity checkpoint | `e544-e543-root-identity1-r2-24s/last.pt` | `outputs/runs/e544-e543-root-identity1-r2-24s/` (local) | 24-step scratch continuation in 40.96s, SHA `3b6e3c00…474f20c`; rank-only identity decode raises OOD `n=4` meaningful 0.00→0.25, structure 0.1250→0.1688, and recall 0.1458→0.2708, but strict meaning 0.0 and AgentV 0/1; no sync or promotion |
| E545 matched negative-weight checkpoints | `e545-e544-root-identity-neg{1-control,4}-r*/last.pt` | `outputs/runs/e545-…/` (local) | Matched 24-step scratch continuations in 30.64s / 28.64s, SHAs `9e54d470…76fa1` / `14dd4404…61ae`; weight 4 slightly improves sparse late negative accuracy, but predictions and OOD `n=4` metrics are identical, both regress from E544, and AgentV is 0/1; no sync or promotion |
| E546 matched strict-subset checkpoints | `e546-e544-strict-subset{1-control,5}-r*/last.pt` | `outputs/runs/e546-…/` (local) | Multiplier 5 raises strict-negative exposure 7→22 rows and improves OOD `n=4` fidelity 0.4250→0.6083, structure 0.1494→0.2038, reward 0.5078→0.8120, and AST edge F1 0→0.0417, but recall falls 0.2083→0.0625, meaning remains 0, AgentV 0/1; no sync or promotion |
| E547 moderate strict-subset checkpoint | `e547-e544-strict-subset2-r1-24s/last.pt` | `outputs/runs/e547-e544-strict-subset2-r1-24s/` (local) | 24-step multiplier-2 scratch run in 36.48s, SHA `37002bfd…0fc57`; OOD `n=4` structure 0.2248 and AST node F1 0.3270 lead the 1/2/5 ladder while recall stays 0.2083, but fidelity falls to 0.2583, meaning remains 0, AgentV 0/1; no sync or promotion |
| E551 no-lexeme-prior checkpoint | `e551-e544-strict-subset2-no-lexeme-r1-24s/last.pt` | `outputs/runs/e551-e544-strict-subset2-no-lexeme-r1-24s/` (local) | 24-step scratch run in 41.85s, SHA `e7921e66…dac32fc6`; fidelity improves to 0.3000, but structure falls to 0.1594 and recall to 0.1250; meaning 0, AgentV 0/1; no sync or promotion |
| E552 half-strength lexeme-prior checkpoint | `e552-e544-strict-subset2-lexeme05-r1-24s/last.pt` | `outputs/runs/e552-e544-strict-subset2-lexeme05-r1-24s/` (local) | 24-step scratch run in 34.75s, SHA `49a9c111…a151fc04`; fidelity 0.1333, structure 0.2181, recall 0.1250, reward 0.3435; meaning 0, AgentV 0/1; no sync or promotion |
| E553 corpus-local proportional-prior checkpoint | `e553-e544-prior-proportional-r3-24s/last.pt` | `outputs/runs/e553-e544-prior-proportional-r3-24s/` (local) | 24-step scratch run in 34.48s, SHA `510e55cf…e75399d`; fidelity 0.3000, structure 0.1244, recall 0.0625, reward 0.5453; meaning 0, AgentV 0/1; no sync or promotion |
| E554 next-slot-context checkpoint | `e554-e544-slot-next-context-r2-24s/last.pt` | `outputs/runs/e554-e544-slot-next-context-r2-24s/` (local) | 24-step scratch run in 39.91s, SHA `af3cbce7…c67b579`; fidelity 0.2583, structure 0.1594, recall 0.1250, reward 0.5328; meaning 0, AgentV 0/1; no sync or promotion |
| E555 slot-pair-interaction checkpoint | `e555-e544-slot-pair-interaction-r2-24s/last.pt` | `outputs/runs/e555-e544-slot-pair-interaction-r2-24s/` (local) | 24-step scratch run in 50.29s, SHA `af53e161…addf19e`; fidelity 0.3000, structure 0.1594, recall 0.1250, reward 0.5453; Pareto lever retained, meaning 0, AgentV 0/1; no sync or promotion |
| E556 combined-slot-context checkpoint | `e556-e544-slot-context-combined-r1-24s/last.pt` | `outputs/runs/e556-e544-slot-context-combined-r1-24s/` (local) | 24-step scratch run in 68.42s, SHA `139c670c…5831f0a`; fidelity 0.2167, structure 0.1594, recall 0.1250, reward 0.5203; combination rejected, meaning 0, AgentV 0/1 |
| E557 full-balance checkpoint | `e557-e544-slot-pair-balance1-r1-24s/last.pt` | `outputs/runs/e557-e544-slot-pair-balance1-r1-24s/` (local) | 24-step scratch run in 70.09s, SHA `438d9871…b97db05`; metrics exactly match E555; no sync or promotion |
| E558 owner-coverage engineering trial | `e558-e544-owner-coverage-r1-24s/last.pt` | `outputs/runs/e558-e544-owner-coverage-r1-24s/` (local) | 24-step scratch run in 43.31s, SHA `8a572738…de85382`; dirty-tree trial persisted but excluded from decisions |
| E558 owner-coverage checkpoint | `e558-e544-owner-coverage-r2-24s/last.pt` | `outputs/runs/e558-e544-owner-coverage-r2-24s/` (local) | 24-step scratch run in 43.74s, SHA `a45909df…381ede`; fidelity 0.4250 but structure/reward regress and AgentV fails; no sync or promotion |
| E559 twofold owner-coverage checkpoint | `e559-e544-owner-coverage2-r1-24s/last.pt` | `outputs/runs/e559-e544-owner-coverage2-r1-24s/` (local) | 24-step scratch run in 31.14s, SHA `1d11926d…9aac861`; fidelity 0.4417 and recall 0.2708, but reward 0.1643 and AgentV fails; no sync or promotion |
| E560 narrow owner-coverage checkpoint | `e560-e544-owner-threshold4-r1-24s/last.pt` | `outputs/runs/e560-e544-owner-threshold4-r1-24s/` (local) | 24-step scratch run in 42.26s, SHA `dae11cee…d7686a3`; structure 0.2181 and AST-node F1 0.3389, but semantic gates fail; no sync or promotion |
| E561 midpoint owner-coverage checkpoint | `e561-e544-owner-threshold7-r1-24s/last.pt` | `outputs/runs/e561-e544-owner-threshold7-r1-24s/` (local) | 24-step scratch run in 41.47s, SHA `35a4fe6d…3a127f9`; fidelity 0.5750, structure 0.2419, reward 0.5753, but meaning/AgentV fail; no sync or promotion |
| E568 design-context continuation checkpoint | `e568-e561-cont48-r1-48s/last.pt` | `outputs/runs/e568-e561-cont48-r1-48s/` (local) | 48-step scratch run in 116.24s, SHA `8dcc0804…0283a12b`; reward 0.6920 but fidelity/structure regress to 0.2583/0.1375 and meaning/AgentV fail; no sync or promotion |
| E569 matched continuation checkpoint | `e569-e561-matched-cont48-r1-48s/last.pt` | `outputs/runs/e569-e561-matched-cont48-r1-48s/` (local) | 48-step scratch run in 75.20s, SHA `8254fcf7…c6535f73`; meaning-v1 0.25, recall 0.3333, reward 0.6920, but strict meaning/AgentV fail; no sync or promotion |
| E572 fidelity-loss checkpoint | `e572-e569-fidelity2-r1-48s/last.pt` | `outputs/runs/e572-e569-fidelity2-r1-48s/` (local) | 48-step scratch run in 84.26s, SHA `bb6a58ff…cc29efa2`; fidelity 0.6500 and reward 0.8170, but meaning-v1/v2 0 and AgentV fails; no sync or promotion |
| E573 midpoint fidelity checkpoint | `e573-e569-fidelity1-r1-48s/last.pt` | `outputs/runs/e573-e569-fidelity1-r1-48s/` (local) | 48-step scratch run in 109.72s, SHA `ff21fc0c…cf59070d`; meaning-v1 0.25, fidelity 0.4750, reward 0.7570, but strict meaning/AgentV fail; no sync or promotion |
| E574 slot-loss checkpoint | `e574-e569-slotloss2-r1-48s/last.pt` | `outputs/runs/e574-e569-slotloss2-r1-48s/` (local) | 48-step scratch run in 76.23s, SHA `649cf512…3810b7c2`; aggregates exactly match E573 and strict meaning/AgentV fail; no sync or promotion |
| CAP5 evidence package | `cap5-03-evidence` | `docs/design/calculated-arity-adaptive-precision-results.md` | Reproducible evidence package for CAP0–CAP4 exact calculations and controlled fixtures; not a checkpoint or ship claim ([results](docs/design/calculated-arity-adaptive-precision-results.md)) |
| Production HF ship | *(none yet)* | [HF Bucket `TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI) `checkpoints/<run_id>/` | Register here after first full HF sync + `--ship-gates` |

**CAP2 capability certificate:** `CERT_CAP2` is not issued. The terminal
SLM-385 ledger records compiler-contract-only symbolic transform and merge
support, rejects learned discrete-token action benefit, and leaves the
CAP1-dependent and conditional branches unavailable or unrun. DSH4 action
distillation is closed; no checkpoint or ship claim changed. See the
[disposition](docs/design/dsh3-17-cap2-disposition-20260723/summary.md).

**Load demo:** `python -m scripts.serve_playground` · **Full train sync:** set
`HF_TOKEN`, then `train_model --context-backend hf` (auto-uploads). Details,
eval tables, and history live in the model card.

## Quick start

```bash
# Node.js 20-22 is required for the locked bridge and browser dependencies.
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"

# Official OpenUI parser + DESIGN.md bridges
cd src/apps/openui_bridge && npm ci && cd ../..
cd src/apps/design_md_bridge && npm ci && cd ../..

# optional MCP server deps
pip install -e ".[mcp]"
# optional live RICO download
pip install -e ".[rico]"
```

## Quick start (train / disjoint test)

Every pipeline phase is also reachable through the unified `slm` CLI
(`slm list` shows the full command map; `slm guide <phase>` prints the
matching operating reference from `.agents/skills/autotrain/references/`). The
`python -m scripts.<name>` forms below remain the direct equivalents.

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild
python -m scripts.build_train_data --source fixture --version v0 --synthesizer quality

# Test suites with strict leakage checks against the train manifest
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json

# Full HF-context trains sync checkpoints to the OpenUI bucket
# (https://huggingface.co/buckets/TKendrick/OpenUI). Requires HF_TOKEN.
export HF_TOKEN=hf_...   # or: hf auth login
python -m scripts.train_model \
  --train-dir outputs/data/train/v1 \
  --model twotower \
  --context-backend hf \
  --steps 200 \
  --run-id twotower_v1
# → hf://buckets/TKendrick/OpenUI/checkpoints/twotower_v1/

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/v1 \
  --model twotower \
  --run-id twotower_v1 \
  --ship-gates
```

### Canonical lever registry

`ModelBuildConfig` is the single source for user-facing model, data, training,
decode, and evaluation lever defaults. Discover the complete machine-readable
set without searching scripts:

```bash
python -m slm_training.levers
python -m slm_training.levers --category decode
```

The catalog identifies intentional checkpoint-vs-harness default differences,
each decode lever's executable tokenizer/compiler configurations, and the
training objective that must exist in a checkpoint before a learned decode
head can be enabled. Invalid, untrained, or inert combinations fail during
config construction or checkpoint override, before run artifacts are made.
The repository-wide run cap is the sole policy lever owned directly by
`src/slm_training/levers.py`; changing `MAX_RUN_MINUTES` updates every Python
consumer. Local compute is the default experiment path. Remote CI and managed
jobs are optional last-resort execution surfaces and are not part of this local
lever registry.

Evaluation uses the [AgentEvals](https://agentevals.io/) JSONL/YAML contract
and the pinned AgentV SDK. Run `npm ci` before Python eval commands; shared
model, loss, task, and diagnostic eval paths automatically write AgentV bundles
beside their domain JSON under `<run-dir>/agentv/`. The existing honest OpenUI
ship gates remain authoritative. See
[the AgentV evaluation contract](docs/design/agentv-evaluation.md).

Local-only / CI scratch: add `--no-sync-checkpoints` (matrix scripts default to
scratch and stay local). Manual sync:
`python -m scripts.sync_checkpoints --run-dir outputs/runs/<id> --ensure-bucket`.
See [docs/design/checkpoint-bucket.md](docs/design/checkpoint-bucket.md).

Checkpoint provenance is fail-closed: each sync emits a verified
`CheckpointReferenceV1`, and `frontier`/`ship_candidate` citations must resolve
from a fresh clone or CI fails (`python -m scripts.verify_checkpoint_references
--check`). See
[docs/design/checkpoint-provenance.md](docs/design/checkpoint-provenance.md).

Honest ship path (V4 inventory-in-prompt / V6 stacked champion):

```bash
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control

# V6: CoRe remask + slot-aware trust + honest V5 alphabet
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control
```

Train artifacts land in `outputs/data/train/<version>/`; eval, preference,
annotation, trajectory, ProgramSpec, and mixture data use sibling typed roots.
Use `slm-data list`, `slm-data resolve train <version>`, and
`slm-data verify train <version>`
instead of memorizing paths. Selected immutable snapshots publish to Git with
`slm-data publish train <version>`.

Every new run writes `outputs/runs/<id>/trace.json` and OTLP JSONL signals under
`outputs/traces/<trace-id>/`. Set `OTEL_EXPORTER_OTLP_ENDPOINT` for an optional
remote OTLP mirror; detailed domain traces remain local and linked by trace ID.
Set `LANGSMITH_TRACING=true` and provide `LANGSMITH_API_KEY` to export an
additional, best-effort aggregate trace to the `slm-training` LangSmith project
(`LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`, and `LANGSMITH_WORKSPACE_ID` are
optional overrides). This exports run/suite metrics, version stamps, gate
verdicts, and AgentV summaries only—never prompts, targets, completions,
checkpoints, raw logs, or credentials. Local evidence and AgentEvals remain
authoritative when LangSmith is unavailable. Install the harness-side client
with `pip install -e '.[observability]'`; the shared trace boundary loads the
repository's ignored `.env` without overriding environment variables provided
by CI or a shell. Standard OTLP settings are honored as well:
`OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`, signal-specific endpoints,
and `OTEL_EXPORTER_OTLP_TIMEOUT` (milliseconds; bounded to five seconds).

The flush pipeline remains: curated seeds + RICO + Awwwards → deterministic
quality synth → per-record DESIGN.md + OpenUI validate → quality gates → stable
sort by `id` + content fingerprint.

Eval uses **meaningful parse** (rejects empty stacks, missing placeholders, and low gold component-type recall), strict `placeholder_fidelity` for ship gates, `structural_similarity`, and composite `reward_score` (does not credit gold DESIGN.md lint). Suites: smoke/held_out (fixtures), `rico_held`, adversarial, ood. Soft `placeholder_validity` is diagnostic only.

**Fixture demo vs ship:** a tiny upsample + scratch + smoke-only fail-under is wiring only. Readiness requires `--ship-gates` on the full scoreboard (see adversarial review).

Expand `rico_held` with 1500 additional HF RICO screens (cached under `src/slm_training/resources/rico/hf_test_cache.jsonl`):

```bash
python -m scripts.build_test_data \
  --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500
```

```bash
# Lightweight unit/integration suite (iterative model training is excluded)
pytest

# Only suites affected by staged + unstaged local changes
.githooks/check-changed

# Repository layout, skill mirrors, and tracked-artifact policy
python -m scripts.repo_policy

# Explicit, compute-intensive model-training tests
pytest -m training
```

Enable the tracked pre-commit hook once per clone with
`git config core.hooksPath .githooks` (a Claude Code `SessionStart` hook arms it
when it is unset). That hook is what runs the changed-file checker.

Agent hooks are narrower and are certified identical across harnesses by
`python -m scripts.verify_agent_surfaces`:

| Harness | Config | Blocks raw `mv` | Post-edit parity, version, and case checks |
| --- | --- | :-: | :-: |
| Claude Code | [`.claude/settings.json`](.claude/settings.json) | yes | yes |
| Codex | [`.codex/hooks.json`](.codex/hooks.json) | yes | yes |
| Copilot CLI | [`.github/hooks/`](.github/hooks/) | yes | yes |
| Cursor, Gemini CLI | — (no hook mechanism configured) | no | no |

Agents on a harness without hooks run `python -m scripts.repo_policy` and
`.githooks/check-changed` themselves. CI remains authoritative either way. See
[`docs/repository-organization.md`](docs/repository-organization.md) and
[`docs/design/agent-harness-parity-audit.md`](docs/design/agent-harness-parity-audit.md).

## OpenUI Lang

Fixtures and validation use official **`openuiLibrary`** syntax, e.g.:

```
root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])
```

Content props must be placeholder strings. Parsing/serialization/prompt generation come from `@openuidev/lang-core` + `@openuidev/react-ui` — see [`src/apps/openui_bridge/`](src/apps/openui_bridge/).

DESIGN.md conditioning + linter: [`src/apps/design_md_bridge/`](src/apps/design_md_bridge/) and [`src/slm_training/resources/design_md/`](src/slm_training/resources/design_md/).

## Mission Control dashboard

`serve_playground` serves a **control-plane + observability SPA** at `/` — one
pane of glass over the whole lifecycle (data → experiments → smoke →
checkpoints/promotion) — including the annotate playground at `/playground`.

```bash
pip install -e ".[dev,torch,web]"
python -m scripts.serve_playground --port 8765        # full control plane (local)
python -m scripts.serve_playground --no-enable-jobs   # read-only observability
# For network exposure, set SLM_ANNOTATION_TOKEN and add --public.
# open http://127.0.0.1:8765
```

Surfaces (React 19 + Vite SPA, dark-first "mission control" design system):

| Route | What |
| --- | --- |
| `/` Overview | Live jobs, experiment scoreboard, checkpoint roster, corpus health, system status, **remote dispatches** |
| `/data` | Navigate + generate versioned corpora (`build_train_data` / `build_test_data`) |
| `/experiments` | Quality / grammar / perf / phase matrices; run `run_*_matrix`; **dispatch bounded GPU checkpoint smokes** (`hf_jobs_train` / `remote_train`); drill into any run |
| `/smoke` | Smoke canary + perf & telemetry; launch wiring runs |
| `/checkpoints` | Roster + **live configurable ship gates** + promote / deploy + blinded A/B |
| `/runs/<id>` | Per-run detail — gate matrix, telemetry spans, `train_summary` metrics, durable-checkpoint link |
| `/playground` | Full annotate UI (React): staged generation, browser fallback/review, DSL repair, and feedback |

**Read vs execute.** Observability views are pure reads (work on a fresh checkout
and on read-only Vercel, falling back to committed `docs/design/*.json` /
`MODEL_CARD.md` / `src/slm_training/resources/`, tagged with `provenance`). Generate/run/promote
actions execute an **allowlisted** set of scripts as tracked background jobs with
live SSE logs — only when served locally (`--enable-jobs`, default on); Vercel
degrades to read-only automatically. Gate math (`POST /api/gates/evaluate`) is
pure, so the threshold editor stays live even read-only. Backend:
`src/slm_training/web/{observability,jobs,capabilities,routes}.py`; SPA source in
[`src/apps/dashboard/`](src/apps/dashboard/) (built bundle committed under
`web/static/app/`, like the preview lib).

**Compiled ↔ interpreted (dogfooding OpenUI).** The sidebar has a
**◈ Compiled / ◇ Interpreted** toggle. *Compiled* is the hand-written React above.
*Interpreted* renders each page from a committed **OpenUI Lang** program
(`src/slm_training/web/static/openui/<slug>.openui`) run **live** through the official
[`@openuidev`](https://openui.com) `<Renderer>` — same components, live `/api` data via a
tool provider, working nav, reactive selectors, launchers, and the live gate editor — so
the app *is* the DSL. The two are kept at parity (`scripts/validate_page_dsl.py` +
`tests/test_web/test_page_dsl.py` + the `dashboard-openui-parity` skill); interpreted-mode
source lives in [`src/apps/dashboard/src/interpret/`](src/apps/dashboard/src/interpret/).

## Annotate playground (`/playground`)

```bash
python -m scripts.serve_playground --port 8765
# open http://127.0.0.1:8765/playground
```

`/playground` is the React annotate UI inside the SPA shell (shares the dark
design system). It owns the complete annotation flow: bounded server attempts,
browser review/fallback, editable and validated DSL corrections, annotator/model
identity, bearer-token support, activity history, keyboard/swipe grading, and the
diffusion progress canvas. The retired `/playground/classic` URL redirects here.
If both model paths are unavailable, the page shows a clearly labeled wiring
fallback so the renderer/editor/annotation flow remains testable; uncorrected
fallback feedback is excluded from derived training data.

The demo checkpoint lives in `src/slm_training/resources/checkpoints/playground_demo/` (committed
`last.pt` + tokenizer + meta). To regenerate it:

```bash
python -m scripts.bootstrap_playground --force
```

If `last.pt` is missing after a sparse checkout, run the bootstrap command above
before starting the playground.
Annotate mode (default UI): auto-generated prompts, prefetch 1–2 samples ahead, and a live **OpenUI visual preview** (same `@openuidev/react-lang` `Renderer` path as [openui.com/demo](https://www.openui.com/demo/github)).

| Input | Action |
|-------|--------|
| `↑` | Thumbs up (persist, stay on sample) |
| `↓` | Thumbs down (persist, stay on sample) |
| `←` / `→` | Previous / next sample |
| typing | Focus optional note |
| swipe | Mobile: horizontal navigate, vertical grade |

Annotations append to `outputs/data/annotation/feedback.jsonl`. Invalid model outputs are quarantined to `outputs/data/annotation/bad_outputs.jsonl` (never shown in the app). Thumbs-up rows promote into `src/slm_training/resources/annotations/human_train.jsonl` (merged by `build_train_data`). Opposite ratings on the same prompt also write `outputs/data/preference/human_pairs.jsonl`.

```bash
python -m scripts.export_annotations status
python -m scripts.export_annotations export
```

### Rebuild the OpenUI preview bundle

```bash
npm run preview:install
npm run preview:build
# writes src/slm_training/web/static/preview/{preview.js,preview.css}
```

### Rebuild the dashboard bundle

```bash
npm run dashboard:install
npm run dashboard:build
# writes src/slm_training/web/static/app/ (built SPA, committed like the preview lib)
```

### Playwright visual / e2e

```bash
npm ci
npx playwright install chromium
# optional agent skills (already in .agents/skills + discovery mirrors)
playwright-cli install --skills
npm run test:e2e
```

MCP (Cursor): [`.cursor/mcp.json`](.cursor/mcp.json) launches `@playwright/mcp`.


- **Context tower**: scratch TokenEncoder **or** frozen HF model (`--context-backend hf`, default `HuggingFaceTB/SmolLM2-135M`)
- **Denoiser tower**: MaskGIT-style masked token prediction with cross-attention to context ([Chang et al. 2022](https://arxiv.org/abs/2202.04200); adapted)
- **Grammar decode**: DFA force-emit + MaskGIT hole-admit + LTR certify so constrained samples stay valid OpenUI ([research lineage](docs/design/research-lineage.md)). Constrained decoding is the product, not a switch — see [decode invariants](docs/design/decode-invariants.md); `--unconstrained-control` (formerly `--no-grammar`) is a diagnostic control arm whose output is never certified or shipped
- **Output tokenizer**: dual-mode — default **compositional** `OpenUITokenizer`, or V5 **lexer / DSL-native** `DSLNativeTokenizer` (`output_tokenizer=lexer`; see [dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md))
- **Eval**: syntax `parse_rate`, separate `meaningful_program_rate`, placeholder fidelity, and canonical tree match — no hidden gold channel at generate time

```bash
# Optional HF context (requires: pip install -e ".[hf]")
python -m scripts.train_model --model twotower --context-backend hf \
  --hf-model HuggingFaceTB/SmolLM2-135M --steps 200 --run-id twotower_hf --fast-train
```

## Hugging Face Jobs (full GPU train)

ZeroGPU Spaces are for short demos only. Full trains use managed Jobs:

```bash
python -m scripts.hf_jobs_train --dry-run --run-id twotower_jobs_v1 --steps 200
# submit: export HF_TOKEN=… && python -m scripts.hf_jobs_train --run-id … --steps 200
```

Details: [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md).

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Agent instructions

All coding agents (Cursor, Claude Code, Codex, Gemini, Copilot / GHCP, …) must
follow **[AGENTS.md](AGENTS.md)**. Canonical skills live in
[`.agents/skills/`](.agents/skills/) (mirrored under `.claude/skills/`,
`.cursor/skills/`, and `.grok/skills/`).

**Iron law:** after any train / eval / bench / profile / telemetry / matrix /
reproduction (or decision-informing ad-hoc) run, update `docs/design/` JSON
**and** the matching measured-results markdown. Full trigger list and recipe
checklist: [AGENTS.md](AGENTS.md) (skill: `documenting-experiment-results`).
Do not leave results only under `outputs/`.

All eval entrypoints also publish standard AgentEvals cases and AgentV SDK
artifacts. Do not add evaluator-specific envelope formats; extend
`src/slm_training/evals/agentv.py`.

### Token-efficiency stack

Repo ships **ponytail**, **caveman**, **headroom**, and **rtk** under
`.agents/skills/` (plus [`RTK.md`](RTK.md), Cursor rules, and GHCP
`.github/copilot-instructions.md`). Details and refresh commands:
[AGENTS.md — Token-efficiency stack](AGENTS.md).

```bash
# RTK binary (once per machine) — must pass `rtk gain`
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
```

### OpenWiki (code mode)

Repository wiki for agents lives under [`docs/openwiki/`](docs/openwiki/) (start at
[`docs/openwiki/quickstart.md`](docs/openwiki/quickstart.md)). Setup uses
[langchain-ai/openwiki](https://github.com/langchain-ai/openwiki) code mode:
[`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) OpenWiki snippets and
[`.github/workflows/openwiki-update.yml`](.github/workflows/openwiki-update.yml).

```bash
npm install -g openwiki@0.1.2
# needs OPENAI_API_KEY (preferred) or OPENROUTER_API_KEY
python -m scripts.update_openwiki --update --print
```

Add repo secret `OPENAI_API_KEY` to enable scheduled OpenWiki update PRs. The
workflow falls back to `OPENROUTER_API_KEY` when OpenAI is unavailable and
fails clearly when neither secret exists. `LANGSMITH_API_KEY` enables optional
tracing.

### Hugging Face CLI + skills

Agents use the official `hf` CLI and the
[huggingface/skills](https://github.com/huggingface/skills) pack (skill:
`hf-cli` plus datasets / papers / trainers / Spaces / … under
[`.agents/skills/`](.agents/skills/)). Cursor also gets the Hugging Face MCP
server via [`.cursor/mcp.json`](.cursor/mcp.json).

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf skills add --force
hf skills update
hf skills add --claude --force
hf skills add --dest=.cursor/skills --force
```

Optional Cursor UI: [marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).
CLI docs: [huggingface_hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Tokens: [settings/tokens](https://huggingface.co/settings/tokens).

### Serena MCP

Semantic code tools via [Serena](https://github.com/oraios/serena) (not
marketplace installs). Project is initialised under [`.serena/`](.serena/);
Cursor / Claude / VS Code MCP configs are wired in-repo. See
[AGENTS.md — Serena MCP](AGENTS.md).

```bash
uv tool install -p 3.13 serena-agent
serena init
serena project health-check
```

## Layout

```
AGENTS.md              # cross-tool agent instructions (required reading)
RTK.md                 # Rust Token Killer usage (shell output compression)
docs/MODEL_CARD.md     # checkpoint roster + eval (README holds a summary)
docs/repository-organization.md # tracked-file placement + move policy
.agents/skills/        # canonical agent skills
src/slm_training/
  dsl/                 # OpenUI adapter + design_md + grammar/{backends,fastpath}
  harnesses/           # train_data, test_data, model_build, rl, preference,
                       # distill, quality(+retrieval), experiments, annotations
  models/              # TwoTower, grammar_diffusion, tokenizers, remask
  data/                # RICO / Awwwards adapters + leakage fingerprints
  evals/               # loss suites / denoising NLL
  runtime/             # accel, telemetry, compression, cactus
  web/                 # mission-control API (observability + jobs) + annotate playground + SPA
src/gpu_multi_farm/    # FastMCP server + farm adapters
src/apps/openui_bridge/   # @openuidev/lang-core Node sidecar
src/apps/design_md_bridge/
src/apps/openui_preview/
scripts/               # CLIs
src/slm_training/resources/              # seed pairs + RICO semantic slices
docs/design/           # architecture + research lineage + contracts
tests/
  test_dsl/            # parser, grammar, design_md
  test_harnesses/      # mirrors harnesses/* (rl is its own suite)
  test_runtime/        # accel / cactus / compression
  test_models/ test_data/ test_web/ ...
```
