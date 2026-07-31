/-
  Decode invariants (I1 / I2 / I6) as a self-contained finite-domain theory.

  Production decode is grammar-constrained end to end. When the certified
  completion domain is a singleton, that symbol is committed with no neural
  ranking. An empty legal domain is a constrained dead end, never a full-
  vocabulary fallback. Rankers may only choose among already-legal candidates.

  Scope: abstract membership and commit rules. Not a proof that a particular
  PyTorch backend implements them — that remains CI + unit tests
  (`forwards_count == 0`, `verify_decode_invariants`).
-/

import LeverProofLean.ListSet

namespace LeverProofLean.DecodeInvariants

open LeverProofLean

abbrev Symbol := Nat

/-- Certified completion domain at one decode step. -/
structure Domain where
  candidates : List Symbol
  /-- Exhaustive coverage is required before reading singleton status. -/
  coverageComplete : Bool
  deriving Repr, BEq

/-- Production config surface for constrained decoding. -/
structure Config where
  grammarConstrained : Bool
  allowUnconstrainedFallback : Bool
  deriving Repr, BEq

/-- Commit outcomes for one decode step. -/
inductive Commit where
  | singletonBypass (token : Symbol)
  | rankedLegal (token : Symbol)
  | deadEnd
  | illegal
  deriving DecidableEq, Repr, BEq

/-- Production configs must stay constrained and fail closed. -/
def productionLegal (cfg : Config) : Prop :=
  cfg.grammarConstrained = true ∧ cfg.allowUnconstrainedFallback = false

/-- True singleton under complete coverage. -/
def isSingleton (domain : Domain) : Prop :=
  domain.coverageComplete = true ∧ domain.candidates.length = 1

def singletonToken? (domain : Domain) : Option Symbol :=
  if domain.coverageComplete then
    match domain.candidates with
    | [token] => some token
    | _ => none
  else
    none

/--
  Commit rule:

  * unconstrained configs are illegal on the production path;
  * empty live domain ⇒ constrained dead end (never full-vocab fallback);
  * complete singleton ⇒ forced bypass with no ranking;
  * otherwise a ranker may pick only a member of the legal domain.
-/
def commit (cfg : Config) (domain : Domain) (rankerTop : Symbol) : Commit :=
  if !cfg.grammarConstrained || cfg.allowUnconstrainedFallback then
    .illegal
  else if domain.candidates.isEmpty then
    .deadEnd
  else
    match singletonToken? domain with
    | some token => .singletonBypass token
    | none =>
        if rankerTop ∈ domain.candidates then
          .rankedLegal rankerTop
        else
          .illegal

/-- I6: production-legal configs never enable unconstrained fallback. -/
theorem production_legal_no_unconstrained (cfg : Config)
    (h : productionLegal cfg) : cfg.allowUnconstrainedFallback = false :=
  h.2

/-- I6: production-legal configs are grammar-constrained. -/
theorem production_legal_constrained (cfg : Config)
    (h : productionLegal cfg) : cfg.grammarConstrained = true :=
  h.1

/-- I2: a complete singleton commits by bypass, ignoring the ranker. -/
theorem singleton_bypasses_ranker
    (cfg : Config) (token rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commit cfg ⟨[token], true⟩ rankerTop = .singletonBypass token := by
  simp [commit, singletonToken?, hprod.1, hprod.2, List.isEmpty]

/-- I6: empty legal domain is a dead end under production configs. -/
theorem empty_domain_is_dead_end
    (cfg : Config) (coverage : Bool) (rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commit cfg ⟨[], coverage⟩ rankerTop = .deadEnd := by
  simp [commit, hprod.1, hprod.2, List.isEmpty]

/-- I6: empty domain never becomes a ranked legal token. -/
theorem empty_domain_never_ranked
    (cfg : Config) (coverage : Bool) (rankerTop token : Symbol)
    (hprod : productionLegal cfg) :
    commit cfg ⟨[], coverage⟩ rankerTop ≠ .rankedLegal token := by
  intro h
  have := empty_domain_is_dead_end cfg coverage rankerTop hprod
  simp [this] at h

/-- Ranker output is accepted only when it already lies in the legal domain. -/
theorem ranked_token_is_legal
    (cfg : Config) (domain : Domain) (rankerTop token : Symbol)
    (h : commit cfg domain rankerTop = .rankedLegal token) :
    token = rankerTop ∧ token ∈ domain.candidates := by
  revert h
  simp only [commit]
  split
  · intro h; cases h
  · split
    · intro h; cases h
    · split
      · intro h; cases h
      · split
        · intro h
          injection h with htok
          subst htok
          constructor
          · rfl
          · assumption
        · intro h; cases h

/-- Unconstrained configs are rejected on the production commit path. -/
theorem unconstrained_is_illegal (domain : Domain) (rankerTop : Symbol) :
    commit ⟨false, false⟩ domain rankerTop = .illegal := by
  simp [commit]

/-- Allowing unconstrained fallback is rejected on the production commit path. -/
theorem unconstrained_fallback_is_illegal (domain : Domain) (rankerTop : Symbol) :
    commit ⟨true, true⟩ domain rankerTop = .illegal := by
  simp [commit]

/-- I1/I2 decision: singleton detection needs complete coverage. -/
theorem incomplete_coverage_not_singleton (domain : Domain)
    (h : domain.coverageComplete = false) :
    singletonToken? domain = none := by
  simp [singletonToken?, h]

/-- Bypass commits only emit a domain member. -/
theorem singleton_bypass_from_token?
    (cfg : Config) (domain : Domain) (rankerTop token : Symbol)
    (hprod : productionLegal cfg)
    (hne : domain.candidates.isEmpty = false)
    (ht : singletonToken? domain = some token) :
    commit cfg domain rankerTop = .singletonBypass token := by
  simp [commit, hprod.1, hprod.2, hne, ht]

theorem singleton_token_mem
    (domain : Domain) (token : Symbol)
    (ht : singletonToken? domain = some token) :
    token ∈ domain.candidates := by
  unfold singletonToken? at ht
  split at ht
  · cases hcands : domain.candidates with
    | nil => simp [hcands] at ht
    | cons t rest =>
        cases rest with
        | nil =>
            simp [hcands] at ht
            subst t
            simp
        | cons _ _ => simp [hcands] at ht
  · cases ht

end LeverProofLean.DecodeInvariants
