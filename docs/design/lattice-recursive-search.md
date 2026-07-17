# Lattice-guided recursive compiler search

**Status:** architecture doc; two independent fixture-grade executions on
2026-07-16. One campaign ran all eight rows through the honest suites (see
[iter-e240-e247-lattice-campaign-20260716.md](iter-e240-e247-lattice-campaign-20260716.md));
a parallel run evaluated E240-E245 on CPU and stopped before E246-E247 because
its predeclared continuation rule failed. No checkpoint was trained, copied,
promoted, or synced, no ship-gate clear is claimed, and the frontier-scale run
on E224+ checkpoints remains open.

## Research question

Can the existing compiler-tree decoder improve valid OpenUI generation by keeping
hard compiler candidates separate from neural preferences, recurring over that
explicit partial-information state, and invoking bounded stochastic search only
when deterministic refinement stalls?

The implementation boundary is intentionally narrow:

- the existing `CompletionForest` is the hard state and remains authoritative;
- model logits rank or perturb compiler-valid branches but never certify them;
- rollback restores an earlier decision, records a local nogood, and rebuilds the
  incremental parser from the stable prefix;
- recurrence reuses the existing TwoTower denoiser rather than introducing a new
  trained HRM/TRM/LDT architecture;
- PTRM- and GRAM-style modes are inference-time trajectory policies over the same
  legal forest, not reproductions of their objectives or benchmark recipes.

This is therefore an **Adapted** compiler-search controller. Paper accuracy,
soundness, compute, and scaling results do not transfer to OpenUI.

## Synthesis from the shared review

The useful decomposition is `hard state + decisions + soft scores`.

| Layer | Repository representation | Invariant |
| --- | --- | --- |
| Hard lattice | compiler-valid `CompletionForest.paths` at a prefix | Candidates may be removed only by compiler evidence or an explicit decision/nogood. |
| Decision trail | chosen semantic branch plus alternatives | A conflict rolls back the most recent implicated decision within a fixed budget. |
| Soft state | denoiser scores and optional seeded perturbation | Scores order legal candidates; they cannot remove the last legal candidate or validate output. |

The controller alternates projection and proposal. Projection asks the existing
incremental grammar/schema/binder machinery for every known legal next action.
Proposal orders those actions. A singleton is committed; a multi-path forest makes
an explicit decision; an empty forest is bottom and triggers rollback. Repeated
state signatures without a smaller candidate set are stagnation, which may trigger
a bounded PTRM-style trajectory. GRAM-style diversity deduplicates terminal
candidates by the repository's AST fingerprint and returns the best validated one.

This preserves the compiler as verifier. A learned conflict or quality score may
prioritize exploration, but cannot create a legal branch, discard all legal
branches, or bypass final OpenUI validation.

### Prefix-global extendability vs. verifier-backed support (VSS0-04)

`CompletionForest` answers **prefix-global extendability**: at this decode
position, a candidate is grammar/schema/binder-admissible and has at least one
maximal forced continuation. That is *not* the same as **support** — whether the
candidate participates in at least one bounded, verifier-accepted completion. The
[`EnumerativeSupportOracle`](../../src/slm_training/dsl/solver/support.py) (VSS0-04)
supplies the latter: it exhausts the candidate's finite completion space and only
returns `UNSUPPORTED` when every branch had `complete` coverage and no completion
verified — a `partial`/`none` forest, a missing verifier capability, or any budget
stop stays `UNKNOWN`, never `UNSUPPORTED`. A forest that admits a candidate as
extendable therefore never authorizes removal; only a replayed support certificate
does. The oracle is not wired into decode, so this section changes no runtime
behavior.

## Campaign design (V9)

The registered runnable rows are hypotheses, not results. The matched controls separate the
value of rollback from stochastic width and semantic diversity.

| ID | Planned condition | Falsifier |
| --- | --- | --- |
| E240 | corrected greedy compiler-tree control | Control cannot reproduce existing valid decode behavior. |
| E241 | hard/soft lattice with bounded rollback | No validity/coverage gain over E240, or termination cost exceeds its guardrail. |
| E242 | stagnation-triggered localized nogoods | Nogoods repeat conflicts or suppress a known valid alternative. |
| E243 | PTRM-style width 4, triggered only at bottom/stagnation | Triggered search does not outperform E241 under matched compute. |
| E244 | PTRM-style width 4, always on | Always-on search is no worse than triggered search; otherwise selective escape is unsupported. |
| E245 | GRAM-style semantic diversity, width 4 | AST deduplication does not increase unique valid candidates. |
| E246 | full stack, width 4 | Combined levers do not beat their strongest component. |
| E247 | full stack, width 8 | Added width does not pay for added verifier/denoiser calls. |

Before execution, the matrix must use the normal honest suites, AgentEvals and
AgentV publication, fixed seeds, identical checkpoint/data lineage, explicit suite
sizes, and the repository ship gates. Required diagnostics include valid/abstain/
invalid rates, compiler coverage, bottoms, rollbacks, nogood hits, stagnation
triggers, trajectories, unique AST candidates, selector regret, denoiser calls,
and wall latency. Any execution must update the quality matrix JSON and markdown;
the implementation itself does not establish a quality gain.

PTRM/GRAM modes fork complete bounded continuations from the last stable compiler
prefix while copying the decision trail and local nogoods. Every terminal candidate
is validated; GRAM mode deduplicates valid terminals by AST fingerprint. Selection
never reads gold and is lexicographic: hard validity, honest slot-contract
satisfaction, model score, then shorter serialization. Telemetry records trigger
reason, recurrence and rollback depth, abstention/exhaustion, verifier calls, valid
trajectories, unique valid ASTs, selector regret, and any invalid-over-valid choice.

## Source inventory and applicability

Metadata was checked against canonical arXiv records on 2026-07-16. Summaries are
paraphrases of the papers' abstracts/method sections. No PDFs are committed.

| Ref | Source | Contribution and OpenUI boundary |
| --- | --- | --- |
| R0 | *Can Tiny Models Actually Reason?*, shared video/discussion | Motivating synthesis only; not academic evidence. Its proposed stack is decomposed and checked against the primary papers below. |
| R1 | Patrick Cousot and Radhia Cousot, “Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints,” POPL 1977, [DOI](https://doi.org/10.1145/512950.512973) | Establishes sound abstraction and lattice fixpoint reasoning. **Adapted:** compiler candidates form a finite partial-information state; this is not a proved Galois connection for all OpenUI programs. |
| R2 | Patrick Cousot, Radhia Cousot, and Laurent Mauborgne, “The Reduced Product of Abstract Domains and the Combination of Decision Procedures,” FoSSaCS 2011, [DOI](https://doi.org/10.1007/978-3-642-19805-2_31) | Relates reduced products of abstract domains to combinations of decision procedures. **Adjacent:** motivates combining compiler domains while keeping hard deductions separate from heuristic ranking; no formal reduced product is reproduced. |
| R3 | Guan Wang et al., “Hierarchical Reasoning Model,” 2025, [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) | Dual-timescale recurrent reasoning. **Adjacent:** no HRM hierarchy or training objective is added. |
| R4 | Zirui Ren and Ziming Liu, “Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models,” 2026, [arXiv:2601.10679](https://arxiv.org/abs/2601.10679) | Questions whether HRM mechanisms match their biological narrative. **Boundary:** reinforces behavior-based ablations instead of architectural labels. |
| R5 | Alexia Jolicoeur-Martineau, “Less is More: Recursive Reasoning with Tiny Networks,” 2025, [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) | Reuses a small shared network for iterative answer/latent refinement. **Adapted:** repeated calls use the existing denoiser and explicit compiler state; no TRM training recipe is claimed. |
| R6 | Amin Sghaier, Ali Parviz, and Alexia Jolicoeur-Martineau, “Probabilistic Tiny Recursive Model,” 2026, [arXiv:2605.19943](https://arxiv.org/abs/2605.19943) | Introduces probabilistic recursive trajectories to escape bad basins. **Adapted:** seeded score perturbations are confined to compiler-valid alternatives and triggered only by bottom/stagnation. |
| R7 | Liam Davis et al., “Lattice Deduction Transformers,” 2026, [arXiv:2605.08605](https://arxiv.org/abs/2605.08605) | Alternates recurrent neural steps with monotone lattice projection and search; trains on its own search states. **Adapted:** hard projection and rollback are implemented, while LDT architecture, alpha supervision, and training remain future empirical work. |
| R8 | Community LDT reconstruction, [GitHub](https://github.com/lcrh/lattice-deduction-transformers) | Code-level comparison target. **Adjacent:** no code or dependency is vendored. The repository describes end-to-end validation as ongoing and discloses corrected Sudoku and maze evaluation details, so it is contextual reproduction evidence rather than a stable author implementation. |
| R9 | Junyeob Baek et al., “Generative Recursive Reasoning,” 2026, [arXiv:2605.19376](https://arxiv.org/abs/2605.19376) | Models multiple valid reasoning outcomes rather than one deterministic answer. **Adapted:** trajectory diversity is measured by validated AST fingerprints, without reproducing GRAM training. |
| R10 | Jacob Austin et al., “Structured Denoising Diffusion Models in Discrete State-Spaces,” 2021, [arXiv:2107.03006](https://arxiv.org/abs/2107.03006) | General discrete diffusion transition processes. **Adjacent:** contextualizes iterative discrete refinement; the compiler controller is not a D3PM objective. |
| R11 | Subham Sekhar Sahoo et al., “Simple and Effective Masked Diffusion Language Models,” 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) | Simplifies absorbing-mask diffusion language modeling. **Existing adapted lineage:** TwoTower training remains unchanged. |
| R12 | Shansan Gong et al., “DiffuCoder,” 2025, [arXiv:2506.20639](https://arxiv.org/abs/2506.20639) | Studies masked diffusion for code and order-flexible generation. **Adjacent:** motivates global revision, not a new checkpoint or objective here. |
| R13 | Yiming Zeng et al., “TreeDiff,” 2025, [arXiv:2508.01473](https://arxiv.org/abs/2508.01473) | Uses AST guidance for diffusion code generation. **Adapted boundary:** OpenUI uses its existing parser/compiler forest and AST fingerprint, not TreeDiff's model. |
| R14 | Tarun Suresh et al., “DINGO,” 2025, [arXiv:2505.23061](https://arxiv.org/abs/2505.23061) | Enforces constraints during diffusion inference. **Adjacent:** supports interleaving constraints and denoising; no DINGO algorithm is reproduced. |
| R15 | Niels Mündler, Jasper Dekoninck, and Martin Vechev, “Constrained Decoding of Diffusion LLMs with Context-Free Grammars,” 2025, [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) | Uses grammar/completion intersection during diffusion decoding. **Existing adapted lineage:** the OpenUI incremental acceptor remains the practical verifier. |
| R16 | Lize Shao et al., “Constrained Code Generation with Discrete Diffusion,” 2026, [arXiv:2605.16829](https://arxiv.org/abs/2605.16829) | Applies program-level constraints to global diffusion states. **Adjacent:** informs whole-state intervention; reported functionality/security results are not inherited. |
| R17 | Shuyin Ouyang et al., “Beyond Execution: Static-Analysis Rewards and Hint-Conditioned Diffusion RL for Code Generation,” 2026, [arXiv:2605.17174](https://arxiv.org/abs/2605.17174) | Uses static-analysis signals for diffusion RL. **Adjacent:** search telemetry may later support rewards, but this campaign does not run RL. |
| R18 | Harshit Joshi et al., “FLAME: A Small Language Model for Spreadsheet Formulas,” 2023, [arXiv:2301.13779](https://arxiv.org/abs/2301.13779) | Domain-specific small-model code generation. **Adjacent:** supports specialization but provides no lattice/search mechanism. |
| R19 | Rohan Bavishi et al., “Neurosymbolic Repair for Low-Code Formula Languages,” 2022, [arXiv:2207.11765](https://arxiv.org/abs/2207.11765) | Combines learned proposals with symbolic repair. **Adapted principle:** neural ordering is subordinate to compiler validation. |
| R20 | Gabriel Poesia et al., “Synchromesh,” 2022, [arXiv:2201.11227](https://arxiv.org/abs/2201.11227) | Constrained semantic decoding with completion engines. **Existing adapted lineage:** `CompletionForest` is the OpenUI-specific completion surface. |
| R21 | Niels Mündler et al., “Type-Constrained Code Generation with Language Models,” 2025, [arXiv:2504.09246](https://arxiv.org/abs/2504.09246) | Adds type inhabitation to constrained decoding. **Adapted boundary:** OpenUI schema, binder, symbol, and slot checks are narrower than TypeScript typing. |
| R22 | Andrew Blinn et al., “Statically Contextualizing Large Language Models with Typed Holes,” 2024, [arXiv:2409.00921](https://arxiv.org/abs/2409.00921) | Communicates expected types and scope at program holes. **Adjacent:** motivates localized failure cones; no ChatLSP protocol is integrated. |
| R23 | Ashwin Kalyan et al., “Neural-Guided Deductive Search for Real-Time Program Synthesis from Examples,” 2018, [arXiv:1804.01186](https://arxiv.org/abs/1804.01186) | Uses a neural model to prioritize deductive search. **Adapted principle:** logits prioritize, while the compiler defines admissibility. |
| R24 | Kevin Ellis et al., “DreamCoder,” 2020, [arXiv:2006.08381](https://arxiv.org/abs/2006.08381) | Wake-sleep program induction and learned abstractions. **Adjacent:** supports iterative synthesis research, but no library learning occurs here. |
| R25 | Emmanuel Bengio et al., “Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation,” 2021, [arXiv:2106.04399](https://arxiv.org/abs/2106.04399) | Samples diverse candidates proportional to reward. **Adjacent:** motivates diversity metrics; GRAM-style inference is not a GFlowNet. |
| R26 | Po-Wei Wang et al., “SATNet,” 2019, [arXiv:1905.12149](https://arxiv.org/abs/1905.12149) | Embeds differentiable satisfiability solving. **Adjacent:** illustrates a hard neural/symbolic boundary; no differentiable SAT layer is introduced. |

## Derivative hypotheses

1. **Projection before preference:** legality and completion coverage improve when
   the compiler forest is materialized before neural ranking, without changing the
   default greedy result on singleton/complete forests.
2. **Localized conflict learning:** a nogood keyed by stable prefix and decision
   prevents exact conflict recurrence while preserving unrelated alternatives.
3. **Selective stochasticity:** bottom/stagnation-triggered perturbation yields at
   least as many unique valid ASTs as always-on perturbation at fewer calls.
4. **Semantic diversity:** AST-fingerprint deduplication outperforms token-string
   deduplication when serialization variants map to the same OpenUI program.
5. **Bounded honesty:** every search terminates by solution, abstention, or an
   explicit backtrack/trajectory budget; invalid output is never returned as valid.

The complete machine-validated R0-R26 inventory records authorship, canonical IDs,
paraphrased summaries, applicability boundaries, and a distinct falsifiable OpenUI
hypothesis per source in
[`lattice-recursive-sources.json`](../../src/slm_training/resources/autoresearch/lattice-recursive-sources.json).

## Measured results (2026-07-16 UTC)

E240-E245 evaluated the unchanged E228 checkpoint on CPU with seed 0, honest
slot contracts, schema and slot-contract context, no DESIGN.md context, and the
five remediated suites (`smoke n=3`, `held_out n=5`, `adversarial n=4`,
`ood n=4`, `rico_held n=3`). The checkpoint SHA-256 was
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a` before
and after the campaign. Every row emitted its suite JSON, AgentEvals JSONL,
AgentV bundle, and `matrix_result.json` under
`outputs/autoresearch/lattice-v9-local/<run_id>/`.

The cells below are `meaningful / structure / component recall`; suite order is
smoke, held-out, adversarial, OOD, RICO held. Parse rate was 1.0 everywhere.

| Rows | Smoke | Held-out | Adversarial | OOD | RICO held | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| E240-E243, E245 | `.333/.464/.250` | `.000/.337/.157` | `.500/.474/.458` | `.000/.375/.208` | `.667/.163/.444` | Identical outputs; four gates fail. Triggered policies never fired. |
| E244 always-on width 4 | `.000/.057/.000` | `.000/.049/.000` | `.000/.054/.000` | `.000/.046/.000` | `.000/.017/.000` | Reject: semantic quality collapsed despite parse validity. |

E244 exercised 76 verifier calls over 19 records, producing 22 valid
trajectories and 22 unique-valid-AST counts (one per selected valid record, not
within-record semantic diversity). It also recorded 81 bottoms, 77 rollbacks,
53 nogoods, 166 nogood hits, four abstentions, and four budget exhaustions.
False hard eliminations, invalid-over-valid selections, and selector regret
were all zero. Its per-suite median latency was 52.4-68.5 seconds, versus
3.4-10.8 seconds for the inert/control rows.

The result falsifies the useful-width premise: always-on trajectory search made
the output syntactically valid but semantically trivial, while bottom/stagnation
triggered search never activated on this checkpoint. E246 and E247 therefore
failed the predeclared requirement for a prior meaningful, structural,
component-recall, or within-record AST-diversity gain without a greater-than-five
point parse/fidelity regression. They remain registered but were intentionally
not executed. The generalized controller and diagnostics are retained as
research infrastructure; no quality, readiness, or ship claim follows.

Machine-readable campaign evidence is in
[`iter-e240-e245-lattice-search-20260716.json`](iter-e240-e245-lattice-search-20260716.json).
