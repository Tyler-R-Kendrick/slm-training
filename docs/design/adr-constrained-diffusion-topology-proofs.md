# Proofs for constrained diffusion × topology (ADR claim cores)

**Status:** machine-checked Lean 4 cores + human map  
**Lean module:** [`src/leverproof_lean/LeverProofLean/ConstrainedDiffusion.lean`](../../src/leverproof_lean/LeverProofLean/ConstrainedDiffusion.lean)  
**Parent ADR:** [adr-constrained-diffusion-topology-split.md](adr-constrained-diffusion-topology-split.md)  
**Replay:** `make -C src/leverproof_lean proofs` (also pulled by `make -C src/leverproof_lean test`)

## What is proved vs not

| Proved (pure law) | Not proved here (measurement / engineering) |
| --- | --- |
| Γ / production only tightens candidate sets | Cold p50 executor ≤ 0.5× V1 (E2) |
| E1 parity = status + multiset equality; no silent UNKNOWN→UNSUPPORTED | Digest load tax ≥10× (E3) |
| Residual ranking never selects from an all-illegal mask | Live Lark ≡ executable DPDA control tables |
| Soft legality is well-formed only when false | Neural quality of topology heads |
| Forced macro edges exclude literal-boundary ids; incomplete ≠ forced edge | AOT table file layout on disk |
| Valid singleton ⇒ forwards optimum 0 | Empirical `forwards_count` on a given checkpoint |
| Honest E9 label is not AOT cold | Warm 89.8× wall-clock samples |
| Stack ranks: executor ≤ circuit; predicate before circuit | Implementation latency |
| Forest-verified draft ⊆ legal domain | Speculative n-gram table freshness |
| Family independence (I14 sketch) | Campaign outcomes |

Lean proves **calculation and set laws**, not sensor truth. Runtime E1/E4/E8 tests still own live OpenUI/Lark evidence.

## Theorem map (idea → statement)

### Family A — Engine executor

| ADR idea | Theorem(s) | Statement (informal) |
| --- | --- | --- |
| Γ leaf filters only tighten | `gamma_only_tightens`, `production_subset_static`, `production_is_gamma_filter` | If `c` is in the Γ-filtered domain, it was already in the static domain; status is preserved |
| Production = static ∩ keep | `production_preserves_status` | Filter does not change declared coverage/status bit |
| E1 multiset parity | `e1_parity_components`, `e1_parity_{reflexive,symmetric,transitive}` | Parity is equality of status and candidate lists (equivalence relation) |
| No silent UNKNOWN→UNSUPPORTED | `no_silent_unknown_to_unsupported` | Multiset-equal domains cannot pair `unsupported` with oracle `incomplete` |
| E4 no literals in forced macros | `forced_macro_no_literals`, `forced_macro_empty`, `forced_macro_append` | Every token in a forced macro is outside the literal-boundary id set; empty is forced; concatenation preserves the property |
| E4 incomplete ≠ forced edge | `forced_edge_requires_complete`, `incomplete_cannot_be_forced_edge` | A forced edge requires complete coverage; incomplete coverage is never a forced edge |
| E8 / I2 singleton zero-forward optimum | `singleton_forwards_optimum_zero`, `singleton_forwards_optimum_is_minimum` | Under a valid complete singleton proof, the optimum forward count is 0 and is minimal |
| Artifact is proof payload | `artifact_not_sole_executor` | Honest artifact role accelerates projection and is **not** the sole execution substrate |
| E9 memo ≠ AOT cold | `e9_honest_not_aot`, `e9_honest_fixture_class` | Honest warm claim is `requestLocalMemoReuse` under `fixtureOrScratch`, not `aotCold` |

### Family B — Diffusion / multi-hole

| ADR idea | Theorem(s) | Statement (informal) |
| --- | --- | --- |
| Rank only inside legal residual | `rankInsideLegal_length_guard`, `rank_single_legal`, `rank_single_illegal`, `rank_skips_illegal_higher_score`, `rank_all_illegal_two`, `legalScored_only_from_true` | Length mismatch → none; illegal-only → none; legal winner preferred over higher illegal score; scored pairs only from true flags |
| Soft legality forbidden | `soft_legality_forbidden`, `residual_authority_is_honest_or_exact` | Well-formed residual support has `softLegality = false` and authority ∈ {exact, honest_overapprox} |
| Honest over-approx | `honest_overapprox_preserves_admit` | If exact admits, over-approx admits (may widen, not shrink admits) |
| Multi-hole *predicate* before circuit | `predicate_before_circuit` | Boolean residual support is stack rank 4; semiring circuit is rank 6 |

### Family C / I3 / I14

| ADR idea | Theorem(s) | Statement (informal) |
| --- | --- | --- |
| Forest-verified draft ⊆ legal | `forest_verified_draft_subset`, `forest_verified_empty_draft`, `forest_verified_cannot_add` | Speculative draft cannot invent candidates outside the legal domain |
| Ordered stack 1→6 | `never_put_six_before_one`, `circuit_has_max_rank`, `executor_has_min_rank`, `ordered_two_respects_rank` | Executor rank 1 ≤ circuit rank 6; every step rank is between them |
| Claim-family independence | `family_independence_example`, `freeze_topology_preserves_executor_pass` | Executor can pass while circuit/topology fail; freezing topology does not force executor fail |

## Human proof sketches (selected)

### Γ only tightens

**Setup.** Static domain \(S = (σ, C)\). Filter \(κ : C \to \{0,1\}\).  
Production \(P = (σ, \{ c \in C \mid κ(c)=1 \})\).

**Claim.** \(c \in P \Rightarrow c \in C\), and status bits equal.

**Proof.** By definition of list filter membership: `c ∈ filter κ C` iff `c ∈ C ∧ κ(c)`. Status field is copied. □  
Formal: `production_subset_static`, `production_preserves_status`.

### Residual ranking ignores illegal scores

**Setup.** Scores \(s_0,\ldots,s_{n-1}\), mask \(m_i \in \{0,1\}\).  
`legalScored` keeps only pairs with \(m_i=1\). Argmax runs only on that list.

**Claim.** If every \(m_i=0\), winner is none. If only index 2 is legal among `[1,9,2]`, winner is 2 even though 9 is larger.

**Proof.** By computation on the defined functions (`rank_all_illegal_two`, `rank_skips_illegal_higher_score`). Membership in `legalScored` implies a true flag was present (`legalScored_only_from_true`). □

### Singleton forwards optimum is zero

**Setup.** Domain complete with exactly one candidate; forced token present.

**Claim.** `ForwardsOptimum = 0` and for all \(f \in \mathbb{N}\), \(0 \le f\).

**Proof.** Unfold definition: the if-guard is true, so result is 0; zero is ≤ any Nat. □  
Runtime E8 still must show a concrete backend achieves this optimum.

### Never put (6) before (1)

**Setup.** `rank(executor)=1`, `rank(circuit)=6`.

**Claim.** \(1 \le 6\).

**Proof.** `decide`. □  
This is the ADR stack law as a ranking obligation, not a schedule algorithm.

### Forest-verified speculation cannot add legality

**Setup.** Draft \(D\), legal \(L\), with \(\forall c \in D.\; c \in L\).

**Claim.** If \(c \notin L\) then \(c \notin D\).

**Proof.** Contrapositive of the subset hypothesis. □

## Executable guards (Lean `#guard`)

```text
rankInsideLegal [1, 9, 2] [true, false, true] == some 2
rankInsideLegal [1, 9] [false, false] == none
ForwardsOptimum { complete, one candidate, forced some } == 0
```

### Advisory residual plane (SFF harness-refined)

Module: `LeverProofLean.AdvisoryResidual` (see also
[semantic-factor-frontier.md](semantic-factor-frontier.md)).

| Harness law | Non-vacuous theorem |
| --- | --- |
| Complete singleton → zero residual work | `singleton_is_zero_work` (from `score` def) |
| Incomplete coverage → not scored | `incomplete_not_scored` |
| filterLegal keys ⊆ legal | `filterLegal_subset` |
| Factor-node membership round-trip | `reconstruct_encode_example` |
| Golden incidence degrees | `golden_degrees` |
| Soft-token collision | `soft_token_collision` |

Bridges: `resources/experiments/semantic_factor_frontier/golden_vectors.v1.json` +
`tests/test_harnesses/experiments/test_semantic_factor_formal.py`.

ADR Domain-level `score_keys_subset` / `singleton_requires_zero_residual_work`
remain lightweight sketches; prefer AdvisoryResidual for harness refinement.

## Relation to runtime tests

| Lean law | Runtime evidence |
| --- | --- |
| Γ / production subset | `static_control_domain.assert_gamma_only_tightens`, E1 parity suites |
| Residual ranking | `residual_support.rank_inside_legal_residual` + tests |
| Forced macro no literals | `walk_static_forced_macro` + E4 tests |
| Singleton forwards optimum 0 | E8 greedy LTR + `collect_decode_stats` |
| E9 honest labels | `completion-kernel-perf-results.json` classification fields |
| Advisory residual control plane | residual scorer unit tests + SFF golden vectors |
| Factor-node membership | `test_factor_node_membership_roundtrip` + Lean encode/reconstruct |

## Rebuild

```bash
make -C src/leverproof_lean proofs
make -C src/leverproof_lean test
```
