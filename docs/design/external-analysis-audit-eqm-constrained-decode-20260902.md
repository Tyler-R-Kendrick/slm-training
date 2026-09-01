# External-analysis audit: EqM report + constrained-decode literature mapping, with one-shot swarm cards (2026-09-02)


Claim class: `analysis` (no train/eval/decode run was performed; every number below is a reproduction from source files or primary papers). `no-bump:` no metric, gate, harness, or matrix file changes. Card S1 in Part E is satisfied by this file; S1 now only needs to add the `adversarial-review.md` link and the MODEL_CARD recount.
Branch: `claude/equilibrium-matching-ebm-analysis-tv4a5r` (repo `Tyler-R-Kendrick/slm-training`).

## Context

Two artifacts were handed in for adversarial review:

1. **The pasted "constrained-diffusion literature is usable" response** (IG-CD, LAVE, Nemotron-TwoTower, FA-constrained dLLM decode, symbol-only v2, and a proposed three-arm decode study).
2. **The attached EqM report** (`eqmatch_critique.agent.final.md` / `.docx`, identical text; the docx footnote file is empty): an adversarial critique of arXiv:2510.02300 plus a comparison to this repo.

Both were checked claim-by-claim against (a) the live repo on this branch, (b) the primary papers and code they cite. The verdict tables below are the evidence base; the hypotheses and swarm cards that follow are derived from what actually held, what drifted, and what was wrong. Nothing in the repo was modified; this file is the only write.

Repo laws that bind every card: `MAX_RUN_MINUTES = 3` (`src/slm_training/levers.py:20`); docs follow every run (`documenting-experiment-results`); `honest-ship-eval` language (fixture-demo vs ship); version stamps (`python -m scripts.verify_version_stamps --check`, bump `src/slm_training/resources/versions.json` or add `no-bump:`); external test cases via `python -m scripts.refresh_test_cases --check --changed`; `python -m scripts.verify_decode_invariants` and `python -m scripts.verify_agent_surfaces` must stay green; `organize-repository` before new tracked paths; never weaken I1–I6; never write a model identifier into commits or files.

---

## Part A — Verdicts on the pasted literature-mapping response

### A.1 The structural problem: it is circular

The response presents IG-CD / LAVE / "intersection-witness" / "LAVE-as-primary rejected" / "forest-verified speculative completion" as *external* corroboration of the repo. Every one of those framings is already authored **in the repo**: `docs/design/decode-invariants.md:21-23` cites IG-CD (arXiv:2508.10111) and LAVE (arXiv:2602.00612); lines 88-99 contain verbatim the "not LAVE-as-primary … sample-N invent complete prefixes from model marginals, then Earley / recovery-rewrite" sentence; `docs/design/adr-constrained-diffusion-topology-split.md:124-135` contains the keep/reject transfer table the response reproduces. The response is a paraphrase of the repo's own ADR sold back as independent confirmation. Only Nemotron (arXiv:2606.26493) and the FA paper (arXiv:2607.07026) are genuinely new to the repo (zero matches for either).

### A.2 Claim ledger

| # | Claim in response | Verdict | Evidence |
|---|---|---|---|
| 1 | IG-CD = Mündler, Dekoninck, Vechev; C++/JSON near-perfect syntax; +1–2% functional avg; +6.9% Dream-7B JSON | **VERIFIED** | arXiv:2508.10111: +1.9% (Con.Γ) / +2.2% (Con.) avg; Dream-7B JSON +6.9%; C++ 99.2–99.7%, JSON 100% syntactic |
| 2 | SMILES ~1.5% functional, +0.2% from constraints | **VERIFIED** | same paper |
| 3 | "That is external evidence for I6: constrained decoding is the product" | **MISLABELED** | I6 is "Never emit invalid grammar" (`decode-invariants.md:191`). "Constrained decoding is the product" is the *section* heading (line 27; `AGENTS.md:53`) |
| 4 | LAVE LLaDA-8B JSON-Bench syntactic@1 99.5 vs IG-CD 94.8 | **VERIFIED** | arXiv:2602.00612v2 |
| 5 | LAVE's transferable piece is "cache-enhanced recovery of rejected holes" | **PARTIAL** | LAVE §3.3 has a *cached-prefix context replacement* triggered after τ consecutive proposal failures. Not hole-specific. Repo already has an analogue: `asap.penalize` at MaskGIT admit-rejects (`twotower.py:15671-15676`) and `remask_ratio` |
| 6 | "keep `SpeculativeRankerV1` over the legal domain … after `root = ` picks `Stack(` at margin 1.0 with no forward" | **STALE** | Reproduced against the shipped table: 27 candidates, pick `Stack(`, margin **1.941**; second example is 26 candidates → `b1` at 15.916, not 25 → `<BIND_1>` at 1.59. Doc (`decode-invariants.md:132-135`) is stale; no test pins the numbers (`tests/test_dsl/test_speculative_rank.py:313-337` asserts only `confident`) |
| 7 | "1,682 certified records" is the n-gram table corpus | **CONTRADICTED** | The committed artifact header says order 4 / 101 sequences / 4,633 tokens / 538 contexts, built from `wf_smoke_v2` (builder default `scripts/build_speculative_ngram_table.py:146-149`). 1,682 is `openui_verified_v1` (`manifest.json core_records`), which the shipped table was **not** built from. `--check` rebuilds with the same defaults so cannot catch this |
| 8 | Speculative ranker is "a working I3 instance" reachable on the decode path | **PATH-CONDITIONAL** | `_select_compiler_path` (`twotower.py:10660`) is called only from `_compiler_ltr_decode_one/_batch` (11767, 12259), which require `compiler_decode_mode != "off"` (default `"off"`, line 551). Production decode is positionwise MaskGIT (`residual-honesty-block-diffusion-20260804.md:59-60`), which never consults the ranker |
| 9 | "Leave the lever default-off until a campaign binds `corpus_fingerprint`" | **VERIFIED** | `speculative_rank: str = "off"` (`twotower.py:788`, `harnesses/model_build/config.py:516`); `decode-invariants.md:137-140` |
| 10 | Nemotron: frozen AR tower, bidirectional denoiser, layer-aligned cross-attn, 30B hybrid, 98.7% / 2.42×, S=16, γ=0.8, first step commits most tokens | **VERIFIED** | arXiv:2606.26493v2: "denoiser layer i attends to context layer i"; "block size S=16", "confidence threshold γ=0.8"; Fig. 4 |
| 11 | "~2.1T tokens (~10% of backbone pretrain)" | **MINOR ERROR** | 2.1T of 25T = 8.4% |
| 12 | "One checkpoint that can run diffusion / mock-AR / AR" | **VERIFIED** (HF model card describes Mock-AR: context tower builds cache, denoiser predicts next token, 1 token/step) | not in the arXiv v2 text the fetch summarised; present on `huggingface.co/nvidia/Nemotron-TwoTower-30B-A3B-Base-BF16` |
| 13 | Repo TwoTower = frozen SmolLM2-135M + width-192 denoiser | **VERIFIED** | `harness_core/lineage/tracks.py:5,16-23` (`TWOTOWER_BASE_ID="HuggingFaceTB/SmolLM2-135M"`, `d_model: 192`, `freeze_context: True`) |
| 14 | Implied: repo cross-attention is the same layer-aligned design | **CONTRADICTED** | One projected `last_hidden_state` (`models/context.py:98,155,197`) broadcast to every denoiser layer (`models/blocks.py:320-333`). Not layer-aligned |
| 15 | "before growing the denoiser, measure whether cross-attention actually moves legal-set decisions" | **GOOD ADVICE, NO INSTRUMENT EXISTS** | zero matches for `context_ablation`/`zero_context`/`shuffled` on a real decode; SLM-166's `linear_shuffled_context` arm is fixture-only with hashed metrics (`slm166_connector_capacity.py:1-5,511`); SLM-218 is weights-only spectral; the real ablation is "NOT RUN (spec only)" (`iter-slm229-looped-latent-differentiation-20260721.md:82`) |
| 16 | I4 "drop singleton rows from the forward, truncate the window at the next grammar checkpoint" | **VERIFIED** (default off) | `decode-invariants.md:143-182`; `runtime/decode_schedule.py` |
| 17 | FA paper (2026): exact inference over DFA/NFA mean-field, 100% satisfaction by construction | **VERIFIED** | arXiv:2607.07026 (Dang & Ermon) |
| 18 | FA paper maps onto `CompletionDomainV1` + I1/I2 | **OVER-REACH** | The paper *explicitly excludes* CFGs ("does not cover richer constraints such as context-free grammars"). `CompletionDomainV1` is a scope-aware LALR/CFG domain. The honest mapping is to the **lexeme-level DFA** in `dsl/grammar/fastpath/engine.py` / `force_emit.py`, not to the CFG domain |
| 19 | "Dream BFCL-Live JSON 63.9% greedy → 22.3% at T=1; constraints recover near-greedy" | **VERIFIED** (substance) | paper: greedy 63.9→71.5 constrained; sampling 22.3→69.0 constrained |
| 20 | "IG-CD → `admit_fill` / hole reparse" as the emptiness engine | **MISLEADING** | `admit_fill` is documented as a **left-prefix over-approximation**, not emptiness (`maskgit_constrain.py:24-35`; ADR amendment 2026-08-04). Exact check is `multi_region_support` (`residual_support.py`), consumed only by the default-off `block_diffusion_decode` lever |
| 21 | "Do not reimplement rustformlang unless a campaign shows `admit_fill` false-admitting a non-completable canvas" | **CONDITION ALREADY MET (negatively)** | Counterexample `root [HOLE] \n )` proven by whole-vocabulary enumeration (`tests/test_dsl/test_residual_support.py`); 442/442 admit probes are suffix-blind on the profile fixture; field FP rate 0/60 on sequential commits only. `rustformlang` does not appear in the repo (it is ETH-SRI's Rust library inside `eth-sri/constrained-diffusion`) |
| 22 | "keep parse/fidelity/request-coverage/structural as separate gates" | **ONE DEAD NAME** | `request_coverage` "had no writer anywhere" and was renamed `contract_recall` (`harness_core/lineage/promotion.py:18-21`; `measurement-honesty-remediation-20260718.md:24,42`); it is not a ship-gate metric. Ship gates (`ship_gates.py:23-62`, `openui_ship_gates_v6.json`) are `meaningful_program_rate`, `structural_similarity`, `component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`, `placeholder_fidelity`, `reward_score`. The "parse" column in `honest-ship-eval` is `meaningful_program_rate`, not `parse_rate` |
| 23 | "`parse_rate` can be 1.0 while meaningful-program rate stays 0" | **VERIFIED** | `eval_runner.py:2372-2378`; `iter-e248-emptiness-probe-20260716.md:10-13` ("syntax parse = 1.0 with meaningful parse ≈ 0") |
| 24 | E236/E237 0/38 choice changes; E729 decode harm; topology frozen pending anti-E237 | **VERIFIED** | `quality-experiment-matrix.md:1323-1336, 5739-5743`; ADR lines 150-151 (E729: meaningful 0.667→0.333) |
| 25 | `OUTPUT_CONTRACT_VERSION=2` symbol-only, NL deferred | **VERIFIED** | `dsl/language_contract.py:34-37` |
| 26 | Endpoints "forwards_count, forced-token fraction" exist | **HALF** | `forwards_count`, `forced_tokens`, `forced_spans`, `speculative_rank_*` exist in `models/decode_stats.py`; **no** `forced_token_fraction` anywhere |
| 27 | `EG_params ≥ 1` rule | **IMPRECISE** | Rule is `EG_params` **LCB** ≥ 1 (`decode-invariants.md:449-450`, `PromotionCriteria.eg_params_lcb_min`), stated in docs, enforced by `promotion_engine.check_parameter_efficiency`; `levers.py` holds only the rationale |
| 28 | `rico_held`, `--ship-gates`, DESIGN.md as ship artifacts | **VERIFIED / PARTIAL** | `ship_gates.py:21` requires `rico_held` (`min_n: 1500`); `--ship-gates` at `scripts/evaluate_model.py:186`; DESIGN.md is the Google `@google/design.md` linter bridge, referenced only by the skill's ship table |
| 29 | A preregistered decode-only campaign with I2 / admit_fill / speculation arms | **DOES NOT EXIST** | no `ExperimentCampaignV1` manifest with such arms; closest is `dsh3-32-…json` `typed_policy_singleton_bypass_on/off` |

### A.3 The proposed three-arm decode study is incoherent as written

- Arm 2 (`admit_fill`) lives on the **MaskGIT** path, gated by `grammar_fastpath_mode ∈ {mask, hybrid}` (`twotower.py:15309-15313`, `admit_on` at 15667).
- Arm 3 (n-gram spans) lives on the **compiler LTR** path (`compiler_decode_mode != "off"`), which never runs `admit_fill`.
- Arm 1 "I2 only" on MaskGIT requires turning admit probes off, which is a legality-weakening configuration and must be labelled a diagnostic control, never a candidate.
- "No weight update" on a fixture checkpoint puts the primary endpoint at a floor: `meaningful_program_rate` on the fixtures is ≈0 already (E224–E236), so "arm 3 does not move meaningful-program rate" would be uninformative. The endpoint must be a decision-level count (legal-set choice flips, forwards saved at matched output) with `meaningful_program_rate` as a *non-regression* gate, not as the discriminating endpoint.
- The study omits the arm that the ADR amendment says has "practical teeth": parallel block commits validated by `multi_region_support`.

Corrected design is Card S5.

### A.4 What survives from the response

Keep: parse/meaningful separation policy (#23), unconstrained arms as diagnostic only (I6), Nemotron as split-existence-proof-not-scale-license, the FA-paper "sampling collapse" citation for "unconstrained arms are diagnostic". Everything else needs the corrections above before it is written anywhere in the repo.

---

## Part B — Verdicts on the EqM report

### B.1 Externally verifiable claims

| # | Report claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Headline FID 1.90 used cfg 1.5 per official README; B/2 row cfg 1.0; paper never mentions guidance | **VERIFIED** | README table rows exactly as quoted; arXiv v3 has zero "guidance"/"CFG" |
| 2 | `sample_gd.py` defaults `--cfg-scale 4.0` | **VERIFIED**; plus a new finding: the released `sample_gd.py` has **no adaptive-stop argument at all** (fixed `--num-sampling-steps 250`), so Algorithm 2's flagship adaptive compute is not in the released sampler |
| 3 | Issue #6: guidance on only 3 of 4 channels | **VERIFIED** (issue body quotes `models.py:296-300`) |
| 4 | Author replies: "legacy code from the SiT paper" (#6), "derivations based on high-dimensional assumption" (#5), "--disp not required nor used" (#13) | **UNVERIFIED** | Issue pages fetched show bodies but no author replies rendered. Card S11 |
| 5 | Table 5: 57.54 / 73.40 / 75.53; Table 8: 2.09 / 3.96 / 3.36; Table 6 OOD rows; Table 9 sFID 4.54 vs 4.49, IS 275.7 vs 277.5; adaptive compute 33.79 vs 32.85 at 40% | **VERIFIED** | arXiv:2510.02300v3 |
| 6 | C.3 writes `∇E(x_k) = −f(x_k)` vs §3.3 `∇E = f` | **VERIFIED as a notational inconsistency**; **OVERCLAIMED** as "proof internally contradictory" — the descent lemma is invariant under a consistent f→−f relabel; the report itself calls it a typo one sentence later |
| 7 | C.2: "the only way the vector-valued expectation can vanish … is if posterior mass concentrates at γ=1" | **VERIFIED** (asserted, not proven, in the paper) |
| 8 | Statement 2 is "false as stated" via antipodal midpoint in d=4096 | **PROVEN as stationary point; OVERCLAIMED as failure mode.** See B.2 |
| 9 | SR-DiT App. B.4 "unable to reproduce … worse than standard flow matching" | **VERIFIED** (quote confirmed; note it was on "SiT-B/1 with representation alignment", i.e. a non-standard config the report omits) |
| 10 | Distance Marching App. C.1: "rescaled flow matching vector field", failed GD/ULA/HMC 2D reproductions, "inconsistent at two corner locations" | **UNVERIFIED** | HTML v1 lacks Appendix C; PDF exceeded fetch limit. Related-work text does say EqM "shows sensitivity to the choice of decaying coefficients with limited theoretical guidance, and its effectiveness drops when switching architectures". Table 1 has 2D EqM numbers (W2 1.433). Card S11 |
| 11 | "Hoover et al. (NFAM @ ICLR 2026) … non-conservative backbone criticism" as external evidence against EqM | **MISFRAMED** | The paper is "Generative Associative Memory via Equilibrium Matching" by Rodriguez, **Hoover**, Guo, **Yilun Du** — co-authored by EqM's senior author, and it reports EqM *working* with a conservative Energy Transformer (FID 28.56 on CIFAR-10). It is an extension, not a critique |
| 12 | Baselines omitted: REPA 1.42, MDTv2 1.58, MAR-H 1.55, RAE 1.51, Dispersive Loss 1.97 (same first author) | **VERIFIED** (standard numbers; Dispersive Loss first author is Runqian Wang) |
| 13 | "LightningDiT at 1.35 **without CFG**" | **LIKELY WRONG** | LightningDiT/VA-VAE reports 1.35 *with* CFG (2.17 without). Card S11 |
| 14 | ICLR 2026: reviews 4 and 8, "absent from the program" | **UNVERIFIED** | OpenReview blocked (challenge page). Not on the orals list; poster status unknown. Card S11 |
| 15 | Semantic Scholar ≈21 citations | **UNVERIFIED** (HTTP 429) |
| 16 | π0-EqM (~10%) and Equilibrium Forcing exist | **VERIFIED** | arXiv:2605.23128 (RoboTwin 40.4→50.2, LIBERO-10 85.2→87.0); arXiv:2608.14706 |
| 17 | Version history v1→v2 NAG pseudocode, v2→v3 one-char loop fix | **UNVERIFIED** (not diffed here; plausible) |

### B.2 The mathematics, re-derived

**The counterexample is correct as far as it goes.** Given `x_γ = γx + (1−γ)ε`, `ε − x = (x_γ − x)/(1−γ)` exactly; at `x̂ = 0` with `x₂ = −x₁`, the Gaussian likelihoods are equal for every γ, so `E[x | x_γ = 0, γ] = 0` and `f(0) = 0` at every d. Proven.

**But the midpoint is a saddle, not a local minimum, and Statement 2 is about local minima.** Along the axis `x̂ = αx₁`, the posterior log-odds for `x₁` vs `x₂` is `∑_γ 2γα‖x₁‖²/(1−γ)²`-weighted, i.e. `E[x|x̂] ≈ tanh(κα)·x₁` with κ ∝ ‖x₁‖² (huge in high d for separated modes). Then `f(αx₁) ∝ (α − tanh(κα))·x₁`, negative for small α>0, so gradient descent `x ← x − ηf` moves *away* from the midpoint toward `x₁`: **E has a local maximum along the data axis** and (by symmetry) a minimum in the d−1 orthogonal directions. Consequences:

- The paper's *definition* ("local minimum … f(x̂)=0") is sloppy (stationarity ≠ minimality), so "false as stated" is technically right but rhetorically inflated.
- Report consequence (ii) "extended near-zero-field regions where GD stalls" is **backwards in high d**: the slope `(κ−1)‖x₁‖` at the saddle is steep, so the `‖f‖ < g_min` basin has vanishing measure; a random init has zero probability of stopping there. The report's own figure ("RMS|f| rises linearly away from the midpoint") shows this.
- Consequence (iii) "provable failure mode of the flagship inference mechanism" is therefore **not established**. It *is* plausible in low d (two-moons), which is where the paper actually fails — the report should have tied its counterexample to the low-d evidence rather than to d=4096.
- The correct strong statement: *for any finite dataset the perfect-training field is available in closed form (mixture posterior + 1-D γ integral), so every EqM theory claim can be tested exactly with no training.* Neither the paper nor the report uses this. Card S7.

**Statement 1 critique** (measure-zero γ=1; density prefactor `(1−γ)^{-d}` diverges) — correct.

**"Perfect training ⇒ memorization theorem" vs Fig. 5** — correct and the strongest point in the report.

### B.3 Repo-facing claims in the report

| # | Report claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Zero hits for 2510.02300 / "Equilibrium Matching" / time-invariant / c(γ) / "energy landscape" / `EqM` | **VERIFIED** on 9,382 tracked files |
| 2 | "equilibrium" in exactly three files | **CONTRADICTED** | six: also `iter-slm229-…20260721.{md,json}` and `research-lineage.md` |
| 3 | `SharedRecursiveDenoiserTower` at `recursive_denoiser.py:194`, recurrence, param-count independence, fixture audit 1/1/2, `RECURSIVE_DEPTH_AUX_MODES` / `intermediate_only`, `ValidatedDepthSupervision.normalized()`, no halting | **VERIFIED** (aux modes live in `twotower.py:1194-1236`, depth fixed at `recursive_denoiser.py:246`) |
| 4 | `CandidateEnergyScorer` "class docstring" quotes | **PARTIAL** — the quotes are the *module* docstring (`solver_energy.py:1-22`); class at line 102 |
| 5 | `legal_edit_flow.py` time-conditioned (`time_encoding="linear"`, `schedule_progress`, `LegalEditRateTargetV1.time`) | **VERIFIED** (`legal_edit_flow.py:31,58`; `flow/targets.py:21`) |
| 6 | `tree_edit_diffusion.py` value-guided beam search, beam 4, arXiv:2405.20519 | **VERIFIED** (`:4-11`, `:1000`) |
| 7 | `closure.py:248 reached_fixed_point` | **PARTIAL** — 248 is the fixed-point docstring; symbol at 192/388 |
| 8 | SLM-282 n=2 negative, seed 1/R=4/`b` 17.487688→17.855324 | **VERIFIED** (`iter-slm282-recurrence-health-20260723.md:75,77`) |
| 9 | Telemetry 0.84→0.26 attributed to seed 1 | **MISATTRIBUTED** — that y-ratio is seed **0** example `a` (0.843641→0.262128); seed 1 runs 0.74/0.80→0.27 |
| 10 | SLM-421: `min_pass_rate=0.5` locked first; seeds 2..21; 18/20; 0.90; Wilson [0.699, 0.972] | **VERIFIED** (`…powered-rerun-20260725.md:12-22`) |
| 11 | SLM-434 `repair_negative` 0/5, LAR3 closed | **VERIFIED** (`docs/brains/repo/recursive-recurrence-health.md:77-82`; note 5 seeds not 20, declared) |
| 12 | "~40 rejected rows in `quality-experiment-matrix.md`" | **CONTRADICTED** — 40 is word occurrences; **2** rows carry a rejected verdict (E249, E252). The real count of rejected checkpoints is in `docs/MODEL_CARD.md`: 64 rows "never reuse/promote/sync/ship" + 1 "never promote/sync/ship" |
| 13 | "a negative/blocked verdict is an explicitly successful closeout" is a MODEL_CARD norm | **MISLOCATED** — sole occurrence is `iter-slm229-looped-latent-differentiation-20260721.md:154` |
| 14 | Repo arXiv citations are the six listed | **CONTRADICTED** — 238 distinct arXiv IDs (six listed all exist; ~2.5% of the set) |
| 15 | d_model=32 fixture; 64,994 vs 74,242 params; "input-space directional gains above 1" | **VERIFIED** (phrase in repo is "from the initial y state"; `iter-rsc-a04-…md:44-45` exists to retract a parity claim: recursive is +14.23% params) |
| 16 | H7 (hide recursion index, decay depth supervision to 0, halt on update norm) | **ADMISSIBLE but under-specified** — adds no params (EG_params trivially passes) but the SLM-434 fixture "leaves no headroom (`ar_only` is hard-valid on every eval example)", so the proposed gate cannot discriminate on that fixture. Card S9 |
| 17 | H9 (dual-head cost-to-go + "distance-to-valid-program" magnitude target) | **DEGENERATE under I6** — every candidate reaching `CandidateRanker` is legal by construction, so the proposed magnitude target is identically zero on the training distribution the head would see. Card S10 refutes numerically |
| 18 | "Under the repo's gates every EqM headline would be an n=1 diagnostic" | **VERIFIED in spirit** — `ExperimentCampaignV1` requires ≥2 seeds for promotion-class manifests (`experiment-campaign-governance.md:72`); anti-E237 requires n≥20 |

### B.4 Overall on the report

Strong: the CFG disclosure gap, the perfect-training memorization contradiction, the Statement-1 measure-zero critique, the baseline omission list, the repo evidence-discipline contrast. Weak: three "proven" items are overclaimed (Statement-2 practical consequence; "internally contradictory"; Hoover-as-critic), four repo-facing counts are wrong (equilibrium files, rejected rows, arXiv IDs, telemetry attribution), and five load-bearing external quotes remain unverified from primary sources. It should not be cited in the repo until Card S11 closes the unverified items and Card S1 records the corrections.

---

## Part C — Rubber-duck cross-cuts

1. **Both documents grade the repo on prose, not on artifacts.** The n-gram provenance drift (A.2 #7) shows the invariant doc can be wrong while `--check` is green because the check compares artifact-to-rebuild, not artifact-to-doc. Any "verified against repo" claim that quotes a doc number needs a reproduction, not a grep.
2. **"Reachable" ≠ "on the production path."** I3 machinery is reachable only via the default-off compiler LTR path. The production MaskGIT loop has no zero-forward ranking mechanism at all beyond I2 singletons.
3. **The only exact multi-hole checker has no default consumer.** `multi_region_support` is consumed only under `block_diffusion_decode=False`. So the IG-CD "emptiness" claim is true of a lever nobody runs.
4. **No instrument exists for the one question both documents agree matters** (does the context tower move legal-set decisions).
5. **EqM's testable content is closed-form.** The perfect-training field for a finite dataset is computable exactly; every "hypothesis" in the report that starts "train EqM-B/2 …" can be pre-screened for pennies.
6. **The FA paper is the closest real analogue of I1/I2, but at the lexeme-DFA level.** Mapping it onto the CFG domain smuggles in a capability the paper disclaims.

---

## Part D — New hypotheses (each with falsifier)

| ID | Hypothesis | Falsifier | Card |
|---|---|---|---|
| N1 | The shipped n-gram table (101 `wf_smoke_v2` sequences) is fixture-overfit: on held_out/adversarial branch points its confident-commit rate and margin distribution will differ materially from a table built from `openui_verified_v1` (1,682) | Margin distributions and pick agreement within noise across suites | S2 |
| N2 | On the production MaskGIT path, the context tower contributes ≈0 to legal-set decisions at fixture scale: zeroing or shuffling context flips <5% of non-singleton legal choices and does not move `meaningful_program_rate` | ≥5% flips or a measurable quality delta | S6 |
| N3 | On grammar-constrained canvases the "first step commits the most tokens" profile (Nemotron Fig. 4) is dominated by **forced** (I2) commits, not confidence commits; splitting the per-step commit histogram by authority shows forced share >50% at step 1 | Confidence-committed share dominates | S12 |
| N4 | Under `block_diffusion_decode=True` the left-prefix `admit_fill` false-admit rate is non-zero (unlike 0/60 sequential), i.e. `block_joint_rejections > 0` on the profile fixture | zero joint rejections across block sizes {2,4,8} | S5 |
| N5 | Adding legal-domain n-gram ranking to the MaskGIT path at positions with ≤8 legal candidates saves ≥20% of forwards at zero decision flips vs intact decode | <20% saved or any flip | S5 (arm M3, spec only) |
| N6 | The EqM midpoint stationary point is a saddle whose `‖f‖<g_min` basin has measure → 0 as d grows; at d=2 it is large enough to trap ≥1% of GD trajectories, at d=4096 <0.01% | basin measure does not shrink with d | S7 |
| N7 | `‖f*(x_i)‖` at training points (exact finite-sample field) decays like d^(−1/2) for fixed mode separation; the two-moons failure is the d=2 end of this curve | flat in d | S7 |
| N8 | The FA-paper exact mean-field DFA posterior, applied at the **lexeme DFA** of `fastpath/engine.py`, would reduce proven-impossible parallel commits to 0 with ≤4% overhead — but only if the repo's lexeme DFA is complete for the canvas language; the hypothesis is that it is **not** (CFG-dependent tokens exist) | lexeme-DFA constraint satisfaction = CFG satisfaction on the fixture | S8 (analysis only) |
| N9 | H9 is ill-posed: the proposed magnitude target has zero variance on the candidate stream | non-zero variance | S10 |
| N10 | H7's halting rule cannot be evaluated on the SLM-434 fixture (no headroom); a fixture with ≥1 hard-invalid `ar_only` example is required before any adaptive-depth campaign | headroom exists | S9 |
| N11 | The EqM report's epistemic-contrast chapter understates the repo record by ~20× in the wrong direction (2 vs 64 rejected verdict rows across the two ledgers); re-counting from `MODEL_CARD.md` strengthens the contrast | counts confirm the report | S1 |
| N12 | The existing MaskGIT `asap.penalize` + `remask_ratio` already cover LAVE's cached-prefix recovery; the measurable gap is only in stall termination (`GenerationExhausted` rate) which is ≈0 on fixtures | non-trivial exhaustion rate | S14 |

---

## Part E — One-shot subagent swarm cards

All cards are independent, own disjoint files, and run in parallel from a fresh clone of the branch. Each subagent reads this section and the files it names; nothing else is assumed. Each card ends by running `python -m scripts.verify_decode_invariants`, `python -m scripts.verify_agent_surfaces`, `python -m scripts.verify_version_stamps --check`, `python -m scripts.refresh_test_cases --check --changed`, and the tests it touches; a card is done only when all are green and its deliverable is committed on `claude/equilibrium-matching-ebm-analysis-tv4a5r` (one commit per card, message prefixed with the card id; no model identifiers). Any train/eval/decode run is wrapped so it cannot exceed `MAX_RUN_MINUTES = 3`; a timed-out run is not evidence and must be reported as such.

### S1 — External-analysis audit ledger (doc only)

**Owns:** `docs/design/external-analysis-audit-eqm-constrained-decode-20260902.md` (new), one link line in `docs/design/adversarial-review.md`.
**Input:** Parts A, B, C of this plan verbatim; the two uploaded external documents (EqM report md/docx; pasted literature-mapping response), not tracked in the repo.
**Deliverable:** A dated audit doc in the repo's adversarial-review style: claim / verdict / evidence pointer tables for both external outputs, the B.2 saddle derivation, the N-hypothesis table, and an explicit "what may be cited from these documents" list. Recount rejected verdict rows from `docs/MODEL_CARD.md` (expected 64+1) and from `quality-experiment-matrix.md` (expected 2) and record the counting method. Frontmatter/claim class: `analysis` (no run). Add `no-bump:` note (no metric/gate/harness change).
**Done when:** doc renders, links resolve, `verify_agent_surfaces` green.

### S2 — Re-provenance the speculative n-gram table and pin its documented numbers

**Owns:** `scripts/build_speculative_ngram_table.py`, `src/slm_training/resources/decode/speculative_ngram_v1.json`, `docs/design/decode-invariants.md` lines 123-141 only, `tests/test_dsl/test_speculative_rank.py`, `src/slm_training/resources/versions.json` (`decode.invariants` component).
**Facts:** the doc says the table is built from `openui_verified_v1` (1,682 records, order 3, 523 contexts) and that `root = ` → `Stack(` at margin 1.0; the artifact header says order 4 / 101 sequences / 4,633 tokens / 538 contexts because the builder default is `WF_SMOKE_V2 if WF_SMOKE_V2.is_file() else CERTIFIED_CORPUS` (`build_speculative_ngram_table.py:146-149`). Measured on the shipped table: `root = ` 27 candidates → `Stack(` margin 1.941; `root = Stack([` 26 → `b1` margin 15.916.
**Deliverable:** make the builder default the certified corpus, train-split only, order as documented; rebuild the artifact; make `--check` also assert the artifact's recorded `corpus_fingerprint` matches the certified corpus manifest (so doc/artifact drift is caught, not just artifact/rebuild drift); rewrite the doc paragraph from *measured* output (candidate counts, picks, margins to 3 decimals, order, contexts, record count); add a test that pins those exact examples (count, pick, margin ±0.01) so the doc can never drift silently again; record both tables' margin distributions on smoke/held_out/adversarial branch points in a short results doc `docs/design/iter-s2-ngram-table-provenance-20260902.md` (tests N1). Bump `decode.invariants`.
**Done when:** `--check` green with the new fingerprint assertion; new test green; doc numbers equal measured; `verify_decode_invariants` green.

### S3 — Retire the dead gate name `request_coverage`

**Owns:** every doc/skill/README line matching `request_coverage|request-coverage` outside `harness_core/lineage/promotion.py` and `measurement-honesty-remediation-20260718.md` (which document the rename); `.agents/skills/honest-ship-eval/SKILL.md` gate table caption.
**Facts:** `promotion.py:18-21` states the name "had no writer anywhere" and is `contract_recall`; the ship-gate "parse" column is `meaningful_program_rate` (`openui_ship_gates_v6.json`).
**Deliverable:** replace stale references with `contract_recall`; add one sentence under the honest-ship-eval gate table stating that the `parse` column is `meaningful_program_rate` and that `parse_rate < 1` on a completed suite is an I6 failure, not a gate. Mirror any changed skill text into `.claude/skills/` if it is a copy rather than a symlink. `no-bump:` if no policy file changes.
**Done when:** `rg request_coverage` returns only the two rename-documenting files; `verify_agent_surfaces` green.

### S4 — Add derived decode fractions to `DecodeStats`

**Owns:** `src/slm_training/models/decode_stats.py`, its tests under `tests/test_models/`, `src/slm_training/web/routes.py:307` surface if it enumerates fields, `versions.json` component for decode telemetry (find the component that stamps `DecodeStats`; if none, `no-bump:`).
**Facts:** fields `forwards_count` (29), `forced_tokens` (101), `forced_spans` (100), `speculative_rank_evaluations/_commits/_tokens/_declined` (105-108), `semantic_singleton_bypasses` (34) exist; no fraction fields exist.
**Deliverable:** add `committed_tokens` if absent and derived read-only properties `forced_token_fraction`, `speculative_commit_fraction`, `forwards_per_committed_token`, all `None` when the denominator is 0 (never a fabricated 0); include them in the serialized dict listed at lines ~1038-1190; unit tests for zero-denominator and a synthetic count; update the field docstring block. No decode behaviour change.
**Done when:** tests green; existing `forwards_count == 0` bypass tests untouched and green.

### S5 — Preregistered decode-only campaign (manifest + fixture execution)

**Owns:** `src/slm_training/resources/campaigns/decode_only_authority_ladder_v1.json` (new; if a different canonical directory exists for `ExperimentCampaignV1` manifests, use it and say so), `docs/design/iter-s5-decode-authority-ladder-20260902.md`, `docs/design/quality-experiment-matrix.md` (append rows only).
**Facts:** contract class `ExperimentCampaignV1` at `src/slm_training/autoresearch/experiment_campaign.py:143-197` (fields `endpoints`, `arms`, `seeds`, `stopping_rules`, `multiplicity_families`, `promotion_gates`, `rollback_gates`, `controls`, `negative_controls`, `mechanism_off_arm_ids`, `executable_kill_criteria`, `claim_class`, `locked_eval_manifest_sha256`). Governance: `docs/design/experiment-campaign-governance.md`. Two decode paths: MaskGIT (production; `admit_fill` gated by `grammar_fastpath_mode ∈ {mask,hybrid}` at `twotower.py:15309`, `admit_on` at 15667; `block_diffusion_decode`/`block_diffusion_block_size` at 763-764 with `multi_region_support` joint check and `block_joint_rejections` counter) and compiler LTR (`compiler_decode_mode`, line 551; ranker via `_select_compiler_path` 10660; levers `speculative_rank`, `speculative_rank_table`, `speculative_rank_margin` 788-790). Profile checkpoint: the `s1_d64` fixture used in `residual-honesty-block-diffusion-20260804.md`. Weakening levers: `levers.CONSTRAINT_WEAKENING_LEVERS`.
**Deliverable:** one locked manifest with two path-scoped ladders. Ladder M (MaskGIT): `M0` admit probes off (**diagnostic negative control**, `mechanism_off_arm_ids`), `M1` default left-prefix admit, `M2` `block_diffusion_decode=True` at block sizes {2,4,8} with exact joint check. Ladder L (compiler LTR, `compiler_decode_mode` on): `L1` singleton-only (`speculative_rank=off`), `L2` `speculative_rank=ngram` at margins {0.5,1.0,2.0}. Endpoints, locked before any run: primary = per-position **legal-choice flips vs the ladder's own control** and forwards saved at matched output; gates = `meaningful_program_rate`, `structural_similarity`, `placeholder_fidelity` non-regression; telemetry = `forwards_count`, `forced_tokens`, `speculative_rank_*`, `block_joint_rejections`, `admit_probe_*`. ≥2 decode seeds, suites smoke/held_out/adversarial, `claim_class="fixture_or_scratch"`, kill criteria (any flip that lowers a gate; any joint rejection under M1 semantics; timeout). Execute every arm on the fixture checkpoint within the 3-minute cap per arm, write the results doc with the iron-law template, append matrix rows, and state plainly that this is fixture evidence, never ship evidence. Tests N4 directly.
**Done when:** manifest validates against `ExperimentCampaignV1`; results doc has every arm's numbers or an explicit timeout row; matrix rows appended; no gate weakened.

### S6 — Context-tower causal ablation on the real decode path

**Owns:** `src/slm_training/models/context.py`, `src/slm_training/models/twotower.py` (a single new config field and its use at the point where `context` is passed into `blocks.py:320-333`), `src/slm_training/levers.py` (register the lever as `diagnostic_only: true`, non-weakening), `tests/test_models/test_context_ablation.py` (new), `docs/design/iter-s6-context-ablation-20260902.md`, matrix rows.
**Facts:** the denoiser receives one projected `last_hidden_state` broadcast to all layers; no `context_ablation`/`zero_context`/shuffle instrument exists on a real decode; SLM-166 arm is fixture-only with hashed metrics; SLM-229 ablation is "NOT RUN (spec only)".
**Deliverable:** lever `context_ablation ∈ {off, zero, shuffle_batch, shuffle_positions}` applied to the projected context tensor at decode time only; a decision-level probe that, for each non-singleton legal domain, records the argmax-over-legal-candidates under intact vs ablated context and counts flips; run on the fixture checkpoint over smoke/held_out/adversarial with ≥2 seeds within the cap; report flip rate, `meaningful_program_rate`/`structural_similarity` deltas, and forwards. Tests N2. Must not touch training or production defaults; must fail closed if the lever is set in a ship-gated config (`require_constrained_production_config`).
**Done when:** test proves `off` is byte-identical to today's decode; lever appears in the catalog with `diagnostic_only`; results doc + matrix rows written.

### S7 — Exact perfect-training EqM field: saddle classification and dimension sweep

**Owns:** `scripts/analysis/eqm_oracle_field.py` (new; confirm placement with `organize-repository`; if `scripts/analysis/` is not canonical, use the directory the skill names), `docs/design/eqm-oracle-field-20260902.md` + `.json`.
**Facts:** for a finite dataset `{x_i}`, forward `x_γ = γx + (1−γ)ε`, `γ~U(0,1)`, the perfectly trained field is `f*(x̂) = E[c(γ)/(1−γ)·(x̂ − x) | x_γ = x̂]`, computable by a 1-D quadrature over γ of a softmax posterior over data points with log-likelihood `−‖x̂ − γx_i‖²/(2(1−γ)²) − d·log(1−γ)`. Use `c_trunc(a=0.8)`, λ=4 (paper default) and `c_linear`. Numpy only; no torch; seconds of compute.
**Deliverable:** (1) reproduce the antipodal-midpoint zero at d ∈ {2, 64, 4096}; (2) finite-difference Hessian of E along the data axis and an orthogonal direction to classify the midpoint (expect max/min = saddle); (3) Monte-Carlo the fraction of Gaussian-init GD trajectories (paper GD, η from Table 2) that terminate under `‖f‖<g_min` within a ball around the midpoint, vs d (N6); (4) `‖f*(x_i)‖/RMS` at training points for a 16-point random dataset vs d (N7); (5) a 3-point equilateral case to look for any spurious *minimum*. Write numbers and plots into the doc; keep the script deterministic (seeded) and under 60 s.
**Done when:** script runs in CI-free mode (`python scripts/analysis/eqm_oracle_field.py --out …`), doc records each number with the falsifier from Part D, `no-bump:`.

### S8 — Literature ledger: honest transfer table for the four new papers

**Owns:** `docs/design/research-lineage.md` (append a dated section only), `docs/design/decode-invariants.md` line 21-23 citation sentence only (add the FA paper), `versions.json` `decode.invariants` bump is **not** needed for a citation line — use `no-bump:`.
**Facts:** repo has zero mentions of arXiv:2606.26493 (Nemotron-TwoTower), 2607.07026 (Dang & Ermon, FA-constrained dLLM; **DFA/NFA only, CFGs excluded**), 2605.23128 (π0-EqM), 2608.14706 (Equilibrium Forcing), 2602.02928 (Distance Marching). Repo already cites 2508.10111 and 2602.00612.
**Deliverable:** for each paper a keep / reject-as-blueprint row in the ADR's style (`adr-constrained-diffusion-topology-split.md:124-131`), naming the repo object it maps to: Nemotron → frozen tower (`tracks.py`), block commits (`block_diffusion_decode`), *not* layer-aligned cross-attn (repo broadcasts one context), scale non-transferable (I16–I19); FA paper → lexeme DFA in `fastpath/engine.py`/`force_emit.py` only, with an explicit note that `CompletionDomainV1` is CFG-scoped and out of the paper's scope (N8 analysis: enumerate on the fixture how many legal-domain decisions are *not* expressible at the lexeme-DFA level); π0-EqM / EqF / Distance Marching → "no repo object; continuous-domain; do not import" with one-line reasons. Also record LAVE's cached-prefix recovery mechanism against `asap.penalize` (`twotower.py:15671-15676`) as "already covered; measure only".
**Done when:** section appended; `verify_agent_surfaces` green.

### S9 — H7 preregistration contract for recursive-tower adaptive depth (manifest only, not run)

**Owns:** `src/slm_training/resources/campaigns/recursive_adaptive_depth_h7_v1.json` (new, same directory decision as S5), `docs/design/iter-s9-h7-contract-20260902.md`.
**Facts:** `SharedRecursiveDenoiserTower` (`recursive_denoiser.py:194`), fixed depth at line 246, telemetry `y_update_norm`/`z_update_norm`/`y_update_state_ratio` on `RecursiveDepthDiagnosticsV1` (858-882) never read back; depth-aux modes at `twotower.py:1194-1236`; SLM-421 protocol (`iter-slm282-recurrence-health-powered-rerun-20260725.md`: `min_pass_rate=0.5` locked first, 20 fresh seeds, Wilson CI); SLM-434 fixture has no headroom (`ar_only` hard-valid on every eval example). EG_params: halting adds zero parameters, so I17 passes trivially; I19 requires size-matched arms.
**Deliverable:** a locked `ExperimentCampaignV1` with arms {fixed-R control, index-hidden + decayed depth supervision, index-hidden + halting on `y_update_state_ratio < τ`}, endpoints (per-depth CE non-regression at every depth, steps-to-halt distribution, wall-clock), ≥20 seeds disjoint from 0..21, kill criteria (any depth CE regression as in SLM-282; halting never triggers; halting triggers at depth 1 on >50% of inputs), and a **fixture-headroom precondition**: the manifest must name a fixture with ≥1 example where `ar_only` is invalid, or declare itself blocked. Do not run.
**Done when:** manifest validates; doc states the headroom precondition and whether any existing fixture satisfies it (search `docs/design/iter-slm434*`, `scripts/run_slm138_recursive_denoiser_fixture.py`).

### S10 — Refute H9 numerically (degenerate magnitude target)

**Owns:** `tests/test_models/test_solver_energy_h9_degeneracy.py` (new), a paragraph appended to `docs/design/external-analysis-audit-eqm-constrained-decode-20260902.md` **only after S1 has landed** — to avoid the file conflict, write the paragraph to `docs/design/iter-s10-h9-degeneracy-20260902.md` instead and let S1's doc link to it.
**Facts:** `CandidateEnergyScorer` (`solver_energy.py:102`) orders "the exact live candidates supplied by the solver's `CandidateRanker` seam"; every candidate is legal by construction (`tree_edit_diffusion.py:4-11`, ADR "never widens legality").
**Deliverable:** a test that walks ≥50 real branch points from the fixture corpus through the `CandidateRanker` seam, computes the report's proposed target ("distance-to-valid-program proxy from the grammar acceptor") for every candidate, and asserts its variance is exactly 0 across candidates at every branch point; the doc records the count and closes H9 as ill-posed under I6, with the one non-degenerate reformulation (distance-to-*gold*-program, which is the existing cost-to-go target).
**Done when:** test green and deterministic.

### S11 — Close the unverified external quotes

**Owns:** `docs/design/external-analysis-audit-eqm-constrained-decode-20260902-addendum.md` (new; separate file so S1 is not blocked).
**Facts:** unverified: Distance Marching App. C.1 quotes ("rescaled flow matching vector field", GD/ULA/HMC 2D failures, "two corner locations"); EqM GitHub author replies on issues #5, #6, #13; ICLR 2026 decision and review scores (OpenReview `xqlxtkOhDX`, blocked by challenge page); Semantic Scholar citation count; LightningDiT 1.35 CFG status (arXiv:2501.01423); arXiv v1→v3 diff of Algorithm 2's loop condition.
**Deliverable:** for each item, a primary-source quote with URL and retrieval date, or "could not retrieve" with the exact blocker. Methods allowed: `WebFetch` on `arxiv.org/pdf/<id>v<n>` split by page range if the fetch limit bites (try `https://arxiv.org/pdf/2602.02928v1` then `export.arxiv.org`), GitHub issue JSON via `https://api.github.com/repos/raywang4/EqM/issues/<n>/comments`, OpenReview via `https://api2.openreview.net/notes?forum=xqlxtkOhDX` and the ICLR virtual site search, Semantic Scholar with backoff. Mark each verdict VERIFIED / CONTRADICTED / UNRETRIEVABLE. No repo code.
**Done when:** every item has a row; the addendum is linked from S1's doc by S1 (S1 adds the link speculatively to the addendum path).

### S12 — Per-step commit-authority histogram on the MaskGIT path

**Owns:** `src/slm_training/models/decode_stats.py` (a per-step list field `step_commits: list[dict]` with `forced`, `confident`, `speculative` counts; coordinate with S4 by touching only a new field block and leaving S4's fraction properties alone — if S4 has landed first, rebase), `tests/test_models/test_step_commit_histogram.py`, `docs/design/iter-s12-commit-authority-profile-20260902.md`.
**Facts:** MaskGIT loop `_generate_maskgit_one` (`twotower.py:15117`), `parallel_unmask="adaptive"` (516-517), forced commits via `_record_exact_bypass` (15305), Nemotron Fig. 4 claim ("first diffusion step commits the largest number of tokens").
**Deliverable:** record per denoising step how many positions were committed and by which authority; run the fixture checkpoint on smoke/held_out with `parallel_unmask ∈ {adaptive, confidence, topk}`; plot/tabulate share-by-authority per step; state whether N3 holds (forced share at step 1 > 50%).
**Done when:** telemetry off-by-default costs nothing (byte-identical outputs), test green, doc written with matrix rows.

### S13 — Docstring/line-reference hygiene surfaced by the audit

**Owns:** `src/slm_training/models/solver_energy.py` (move the authority sentences into the class docstring as well), `src/slm_training/dsl/solver/closure.py` (no code change; ensure `reached_fixed_point` has a docstring line so a grep for the symbol lands on its definition), `docs/brains/repo/recursive-recurrence-health.md` (attribute the 0.84→0.26 y-ratio to seed 0 example `a` explicitly, keep seed 1 numbers).
**Deliverable:** three tiny edits, `no-bump:`.
**Done when:** tests untouched and green.

### S14 — Stall/exhaustion telemetry vs LAVE recovery

**Owns:** `docs/design/iter-s14-exhaustion-rate-20260902.md`; read-only use of existing counters (`admit_probe_canvases`, `asap` penalties, `GenerationExhausted` in `web/service.py`).
**Facts:** LAVE recovers from τ consecutive proposal failures by replacing the context with a cached prefix; the repo penalizes rejected candidates (`asap.penalize`) and can remask (`remask_ratio`, default 0.0).
**Deliverable:** on the fixture checkpoint across smoke/held_out/adversarial, count per document: admit rejections, consecutive-rejection run lengths, and decodes ending in exhaustion or certified substitution; report the max consecutive-rejection run and the exhaustion rate; conclude whether N12 holds (rate ≈ 0 ⇒ LAVE recovery is not a gap) and, if not, propose the τ-analogue as a preregistered lever without implementing it.
**Done when:** doc written with numbers, matrix rows appended.

### Ownership matrix (conflict check)

| File | Cards |
|---|---|
| `docs/design/decode-invariants.md` | S2 (lines 123-141), S8 (line 21-23 only) — disjoint hunks |
| `models/decode_stats.py` | S4 (fraction properties), S12 (per-step field) — disjoint blocks; S12 rebases if later |
| `models/twotower.py` | S6 only |
| `quality-experiment-matrix.md` | S5, S6, S12, S14 append rows — append-only, rebase on conflict |
| `versions.json` | S2 only (others `no-bump:`) |
| everything else | single owner |

---

## Verification (orchestrator, after all cards land)

1. `python -m scripts.verify_decode_invariants && python -m scripts.verify_agent_surfaces && python -m scripts.verify_version_stamps --check && python -m scripts.refresh_test_cases --check --changed`.
2. `pytest tests/test_dsl/test_speculative_rank.py tests/test_models -q -k "context_ablation or step_commit or h9 or decode_stats"`.
3. `python scripts/build_speculative_ngram_table.py --check` (S2 fingerprint assertion).
4. `python scripts/analysis/eqm_oracle_field.py` reproduces the numbers in `docs/design/eqm-oracle-field-20260902.md`.
5. Read `docs/design/external-analysis-audit-eqm-constrained-decode-20260902.md` and its addendum: every UNVERIFIED row in Part B has a VERIFIED / CONTRADICTED / UNRETRIEVABLE disposition.
6. Confirm no production default changed: `git diff origin/main -- src/slm_training/harnesses/model_build/config.py` shows no lever default flipped; new levers carry `diagnostic_only: true`.
7. Push branch, open one PR titled "Audit of external EqM/constrained-decode analyses + decode-authority evidence" listing card ids; subscribe to PR activity.
