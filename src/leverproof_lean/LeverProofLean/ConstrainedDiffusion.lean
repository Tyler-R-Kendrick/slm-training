/-
  Constrained diffusion × topology — formal claim cores.

  Pure set / ranking / status laws behind
  `docs/design/adr-constrained-diffusion-topology-split.md`.

  Not claimed: empirical cold speedups, neural quality, live Lark = DPDA.
  Closed development only (Makefile forbids open proof escapes).
-/

namespace LeverProofLean
namespace ConstrainedDiffusion

abbrev Candidate := List Nat

inductive DomainStatus where
  | complete
  | incomplete
  | unsupported
deriving DecidableEq, Repr, Inhabited

structure Domain where
  status : DomainStatus
  candidates : List Candidate
deriving Repr, Inhabited

namespace Domain

def MultisetEq (a b : Domain) : Prop :=
  a.status = b.status ∧ a.candidates = b.candidates

def Has (d : Domain) (c : Candidate) : Prop := c ∈ d.candidates

end Domain

/-! ### Stack-1 / I6 — Γ leaf filters only tighten -/

def GammaFilter (static filtered : Domain) : Prop :=
  filtered.status = static.status ∧
  ∀ c, Domain.Has filtered c → Domain.Has static c

theorem gamma_only_tightens
    (static filtered : Domain) (h : GammaFilter static filtered)
    (c : Candidate) (hc : Domain.Has filtered c) :
    Domain.Has static c :=
  h.2 c hc

def productionDomain (static : Domain) (keep : Candidate → Bool) : Domain :=
  { status := static.status
    candidates := static.candidates.filter keep }

theorem production_subset_static
    (static : Domain) (keep : Candidate → Bool) (c : Candidate)
    (hc : Domain.Has (productionDomain static keep) c) :
    Domain.Has static c := by
  simp [Domain.Has, productionDomain, List.mem_filter] at hc
  exact hc.1

theorem production_preserves_status (static : Domain) (keep : Candidate → Bool) :
    (productionDomain static keep).status = static.status :=
  rfl

theorem production_is_gamma_filter (static : Domain) (keep : Candidate → Bool) :
    GammaFilter static (productionDomain static keep) := by
  refine ⟨production_preserves_status static keep, ?_⟩
  intro c hc
  exact production_subset_static static keep c hc

/-! ### E1 — parity = status + multiset equality; no silent UNKNOWN collapse -/

theorem e1_parity_components
    (production oracle : Domain) (h : Domain.MultisetEq production oracle) :
    production.status = oracle.status ∧
      production.candidates = oracle.candidates :=
  h

theorem no_silent_unknown_to_unsupported
    (production oracle : Domain) (h : Domain.MultisetEq production oracle) :
    ¬ (production.status = .unsupported ∧ oracle.status = .incomplete) := by
  intro ⟨hp, ho⟩
  have := h.1
  simp [hp, ho] at this

theorem e1_parity_symmetric
    (a b : Domain) (h : Domain.MultisetEq a b) : Domain.MultisetEq b a :=
  ⟨h.1.symm, h.2.symm⟩

theorem e1_parity_reflexive (d : Domain) : Domain.MultisetEq d d :=
  ⟨rfl, rfl⟩

theorem e1_parity_transitive
    (a b c : Domain)
    (hab : Domain.MultisetEq a b) (hbc : Domain.MultisetEq b c) :
    Domain.MultisetEq a c :=
  ⟨hab.1.trans hbc.1, hab.2.trans hbc.2⟩

/-! ### Stack-3 — rank only inside the legal residual -/

/-- Indexed score list: `[(0,s0), (1,s1), ...]`. -/
def enumerateScores : List Int → Nat → List (Nat × Int)
  | [], _ => []
  | s :: ss, i => (i, s) :: enumerateScores ss (i + 1)

/-- Keep only pairs whose legal flag is true (parallel lists). -/
def legalScored : List (Nat × Int) → List Bool → List (Nat × Int)
  | [], _ => []
  | _, [] => []
  | (i, s) :: rest, true :: flags => (i, s) :: legalScored rest flags
  | _ :: rest, false :: flags => legalScored rest flags

def pickMaxIndex : List (Nat × Int) → Option Nat
  | [] => none
  | p :: ps =>
      some (ps.foldl (fun best cur => if cur.2 > best.2 then cur else best) p).1

def rankInsideLegal (scores : List Int) (legal : List Bool) : Option Nat :=
  if scores.length ≠ legal.length then none
  else pickMaxIndex (legalScored (enumerateScores scores 0) legal)

theorem rankInsideLegal_length_guard
    (scores : List Int) (legal : List Bool)
    (h : scores.length ≠ legal.length) :
    rankInsideLegal scores legal = none := by
  simp [rankInsideLegal, h]

theorem rank_empty_inputs :
    rankInsideLegal ([] : List Int) ([] : List Bool) = none := by
  simp [rankInsideLegal, enumerateScores, legalScored, pickMaxIndex]

theorem rank_single_illegal :
    rankInsideLegal [0] [false] = none := by
  simp [rankInsideLegal, enumerateScores, legalScored, pickMaxIndex]

theorem rank_single_legal :
    rankInsideLegal [7] [true] = some 0 := by
  simp [rankInsideLegal, enumerateScores, legalScored, pickMaxIndex]

theorem rank_skips_illegal_higher_score :
    rankInsideLegal [1, 9, 2] [true, false, true] = some 2 := by
  simp [rankInsideLegal, enumerateScores, legalScored, pickMaxIndex]

theorem rank_all_illegal_two :
    rankInsideLegal [1, 9] [false, false] = none := by
  simp [rankInsideLegal, enumerateScores, legalScored, pickMaxIndex]

/-- Every pair emitted by `legalScored` came from a `true` flag slot. -/
theorem legalScored_only_from_true
    (pairs : List (Nat × Int)) (flags : List Bool)
    (i : Nat) (s : Int)
    (h : (i, s) ∈ legalScored pairs flags) :
    true ∈ flags := by
  induction pairs generalizing flags with
  | nil =>
      cases flags <;> simp [legalScored] at h
  | cons p ps ih =>
      match flags with
      | [] => simp [legalScored] at h
      | f :: fs =>
          cases f with
          | false =>
              simp [legalScored] at h
              exact List.mem_cons_of_mem false (ih fs h)
          | true =>
              simp [legalScored] at h
              cases h with
              | inl heq =>
                  exact List.mem_cons_self
              | inr hrest =>
                  exact List.mem_cons_of_mem true (ih fs hrest)

/-! ### Soft legality forbidden; honest over-approx -/

structure ResidualSupport where
  admitted : Bool
  authority : String
  softLegality : Bool
deriving Repr

def ResidualSupport.wellFormed (r : ResidualSupport) : Prop :=
  r.softLegality = false ∧
  (r.authority = "exact" ∨ r.authority = "honest_overapprox")

theorem soft_legality_forbidden
    (r : ResidualSupport) (hw : ResidualSupport.wellFormed r) :
    r.softLegality = false :=
  hw.1

theorem residual_authority_is_honest_or_exact
    (r : ResidualSupport) (hw : ResidualSupport.wellFormed r) :
    r.authority = "exact" ∨ r.authority = "honest_overapprox" :=
  hw.2

def HonestOverapprox (exact over : ResidualSupport) : Prop :=
  ResidualSupport.wellFormed exact ∧
  ResidualSupport.wellFormed over ∧
  over.authority = "honest_overapprox" ∧
  exact.authority = "exact" ∧
  (exact.admitted = true → over.admitted = true)

theorem honest_overapprox_preserves_admit
    (exact over : ResidualSupport) (h : HonestOverapprox exact over)
    (ha : exact.admitted = true) :
    over.admitted = true :=
  h.2.2.2.2 ha

/-! ### E4 — forced macro never contains literal-boundary tokens -/

def isLiteralBoundary (tok : Nat) (literalIds : List Nat) : Prop :=
  tok ∈ literalIds

def ForcedMacro (tokens : List Nat) (literalIds : List Nat) : Prop :=
  ∀ t, t ∈ tokens → ¬ isLiteralBoundary t literalIds

theorem forced_macro_no_literals
    (tokens literalIds : List Nat) (h : ForcedMacro tokens literalIds)
    (t : Nat) (ht : t ∈ tokens) :
    t ∉ literalIds := by
  have := h t ht
  simpa [isLiteralBoundary] using this

theorem forced_macro_empty (literalIds : List Nat) :
    ForcedMacro [] literalIds := by
  intro t ht
  cases ht

theorem forced_macro_append
    (a b literalIds : List Nat)
    (ha : ForcedMacro a literalIds) (hb : ForcedMacro b literalIds) :
    ForcedMacro (a ++ b) literalIds := by
  intro t ht
  rcases List.mem_append.mp ht with h | h
  · exact ha t h
  · exact hb t h

inductive Coverage where
  | complete
  | incomplete
deriving DecidableEq, Repr

def ForcedEdge
    (tokens : List Nat) (coverage : Coverage) (literalIds : List Nat) : Prop :=
  coverage = .complete ∧ ForcedMacro tokens literalIds

theorem forced_edge_requires_complete
    (tokens : List Nat) (coverage : Coverage) (literalIds : List Nat)
    (h : ForcedEdge tokens coverage literalIds) :
    coverage = .complete :=
  h.1

theorem incomplete_cannot_be_forced_edge
    (tokens literalIds : List Nat) :
    ¬ ForcedEdge tokens .incomplete literalIds := by
  intro h
  cases h.1

/-! ### E8 / I2 — singleton complete domain ⇒ forwards optimum 0 -/

structure SingletonProof where
  domain : Domain
  forcedToken : Option Nat
deriving Repr

def SingletonProof.valid (p : SingletonProof) : Prop :=
  p.domain.status = .complete ∧
  p.domain.candidates.length = 1 ∧
  p.forcedToken.isSome = true

def ForwardsOptimum (p : SingletonProof) : Nat :=
  if p.domain.status = .complete ∧ p.domain.candidates.length = 1 then 0 else 1

theorem singleton_forwards_optimum_zero
    (p : SingletonProof) (hv : SingletonProof.valid p) :
    ForwardsOptimum p = 0 := by
  rcases hv with ⟨hs, hc, _⟩
  simp [ForwardsOptimum, hs, hc]

theorem singleton_forwards_optimum_is_minimum
    (p : SingletonProof) (hv : SingletonProof.valid p) (f : Nat) :
    ForwardsOptimum p ≤ f := by
  have h0 := singleton_forwards_optimum_zero p hv
  rw [h0]
  exact Nat.zero_le f

/-! ### E9 — request-local memo reuse is not AOT cold -/

inductive PerfClaimClass where
  | fixtureOrScratch
  | ship
deriving DecidableEq, Repr

inductive WarmLabel where
  | requestLocalMemoReuse
  | aotCold
deriving DecidableEq, Repr

structure WarmSpeedupClaim where
  claimClass : PerfClaimClass
  label : WarmLabel
  notAotCold : Bool
deriving Repr

def WarmSpeedupClaim.honest (c : WarmSpeedupClaim) : Prop :=
  c.claimClass = .fixtureOrScratch ∧
  c.label = .requestLocalMemoReuse ∧
  c.notAotCold = true

theorem e9_honest_not_aot
    (c : WarmSpeedupClaim) (h : WarmSpeedupClaim.honest c) :
    c.label ≠ .aotCold := by
  intro heq
  have := h.2.1
  simp [this] at heq

theorem e9_honest_fixture_class
    (c : WarmSpeedupClaim) (h : WarmSpeedupClaim.honest c) :
    c.claimClass = .fixtureOrScratch :=
  h.1

/-! ### Ordered stack — never put (6) before (1) -/

inductive StackStep where
  | executorStaticGamma
  | forcedMacros
  | choiceCodec
  | multiHolePredicate
  | legalReweight
  | semiringCircuit
deriving DecidableEq, Repr

def StackStep.rank : StackStep → Nat
  | .executorStaticGamma => 1
  | .forcedMacros => 2
  | .choiceCodec => 3
  | .multiHolePredicate => 4
  | .legalReweight => 5
  | .semiringCircuit => 6

def OrderedPlan : List StackStep → Prop
  | [] => True
  | [_] => True
  | a :: b :: rest => a.rank ≤ b.rank ∧ OrderedPlan (b :: rest)

theorem ordered_plan_nil : OrderedPlan [] := trivial

theorem ordered_plan_singleton (s : StackStep) : OrderedPlan [s] := trivial

theorem circuit_has_max_rank (s : StackStep) :
    s.rank ≤ StackStep.rank .semiringCircuit := by
  cases s <;> decide

theorem executor_has_min_rank (s : StackStep) :
    StackStep.rank .executorStaticGamma ≤ s.rank := by
  cases s <;> decide

theorem never_put_six_before_one :
    StackStep.rank .executorStaticGamma ≤ StackStep.rank .semiringCircuit := by
  decide

theorem ordered_two_respects_rank
    (a b : StackStep) (h : OrderedPlan [a, b]) :
    a.rank ≤ b.rank := by
  simpa [OrderedPlan] using h.1

/-! ### Claim-family separation (I14) -/

inductive ClaimFamily where
  | engineExecutor
  | diffusionCircuit
  | modelFacingTopology
deriving DecidableEq, Repr

def FamilyResults := ClaimFamily → Bool

theorem family_independence_example
    (r : FamilyResults)
    (hA : r .engineExecutor = true)
    (hB : r .diffusionCircuit = false) :
    r .engineExecutor = true ∧ r .diffusionCircuit = false :=
  ⟨hA, hB⟩

theorem freeze_topology_preserves_executor_pass
    (r : FamilyResults)
    (hExec : r .engineExecutor = true) :
    r .engineExecutor = true :=
  hExec

/-! ### I3 forest-verified speculation — draft ⊆ legal domain -/

def SpeculativeDraft (legal : List Candidate) (draft : List Candidate) : Prop :=
  ∀ c, c ∈ draft → c ∈ legal

theorem forest_verified_draft_subset
    (legal draft : List Candidate) (h : SpeculativeDraft legal draft)
    (c : Candidate) (hc : c ∈ draft) :
    c ∈ legal :=
  h c hc

theorem forest_verified_empty_draft (legal : List Candidate) :
    SpeculativeDraft legal [] := by
  intro c hc
  cases hc

theorem forest_verified_cannot_add
    (legal draft : List Candidate)
    (h : SpeculativeDraft legal draft)
    (c : Candidate) (hc : c ∉ legal) :
    c ∉ draft := by
  intro hin
  exact hc (h c hin)

/-! ### Multi-hole residual predicate vs circuit (paper bound) -/

inductive MultiHoleClaim where
  | booleanPredicate
  | semiringCircuit
deriving DecidableEq, Repr

def MultiHoleClaim.stackStep : MultiHoleClaim → StackStep
  | .booleanPredicate => .multiHolePredicate
  | .semiringCircuit => .semiringCircuit

theorem predicate_before_circuit :
    MultiHoleClaim.booleanPredicate.stackStep.rank <
      MultiHoleClaim.semiringCircuit.stackStep.rank := by
  decide

/-! ### Artifact is proof payload, not free execution substrate -/

structure ArtifactRole where
  acceleratesProjection : Bool
  isSoleExecutionSubstrate : Bool
deriving Repr

def ArtifactRole.honest (r : ArtifactRole) : Prop :=
  r.acceleratesProjection = true ∧ r.isSoleExecutionSubstrate = false

theorem artifact_not_sole_executor
    (r : ArtifactRole) (h : ArtifactRole.honest r) :
    r.isSoleExecutionSubstrate = false :=
  h.2

/-! ### Advisory residual: legal support preservation + singleton zero-work -/

/-- Advisory score map may only key existing legal candidates. -/
def ScoreKeysLegal (legal : List Candidate) (keys : List Candidate) : Prop :=
  ∀ k, k ∈ keys → k ∈ legal

theorem score_keys_subset
    (legal keys : List Candidate) (h : ScoreKeysLegal legal keys)
    (k : Candidate) (hk : k ∈ keys) :
    k ∈ legal :=
  h k hk

/-- Complete singleton forbids residual/neural work (counts are Nat optima). -/
structure ResidualWorkCounts where
  factorCalls : Nat
  propagationCalls : Nat
  rankerCalls : Nat
  neuralForwards : Nat
deriving Repr

def ResidualWorkCounts.zeroWork (w : ResidualWorkCounts) : Prop :=
  w.factorCalls = 0 ∧
  w.propagationCalls = 0 ∧
  w.rankerCalls = 0 ∧
  w.neuralForwards = 0

def SingletonComplete (domain : Domain) : Prop :=
  domain.status = .complete ∧ domain.candidates.length = 1

theorem singleton_requires_zero_residual_work
    (domain : Domain) (w : ResidualWorkCounts)
    (hs : SingletonComplete domain)
    (hpolicy : SingletonComplete domain → ResidualWorkCounts.zeroWork w) :
    ResidualWorkCounts.zeroWork w :=
  hpolicy hs

/-! ### Soft-token non-injectivity (SHIFT correction, finite model) -/

/-- Distinct simplex points may collide under a rank-deficient embedding map.

Model: two distinct Nat-coded distributions with equal weighted embedding sum
when the embedding dimension is smaller than the free simplex dimension.
We prove a concrete 3-vocab / 1-dim counterexample: p=(1,0,0) and q=(0,1,0)
collide when E maps both first two basis vectors to the same scalar.
-/
def softToken (weights : List Nat) (embed : List Nat) : Nat :=
  ((weights.zip embed).foldl (fun acc pair => acc + pair.1 * pair.2) 0)

theorem soft_token_collision_example :
    softToken [1, 0, 0] [7, 7, 3] = softToken [0, 1, 0] [7, 7, 3] ∧
    [1, 0, 0] ≠ [0, 1, 0] := by
  decide

/-! ### Restart contraction rate bound (HyCE operator, abstract) -/

/-- ℓ₁ contraction factor for T(c)=λr+(1-λ)Sc when ‖S‖₁≤1 is at most (1-λ). -/
def restartContractionFactor (lambdaRestart : Nat) (denom : Nat) : Nat :=
  -- Represent λ = lambdaRestart / denom with 0 < λ ≤ 1 as integers.
  denom - lambdaRestart

theorem restart_contraction_factor_le_one
    (lambdaRestart denom : Nat)
    (hpos : 0 < lambdaRestart)
    (hle : lambdaRestart ≤ denom)
    (hdenom : 0 < denom) :
    restartContractionFactor lambdaRestart denom < denom := by
  have : denom - lambdaRestart < denom := Nat.sub_lt hdenom hpos
  simpa [restartContractionFactor] using this

end ConstrainedDiffusion
end LeverProofLean
