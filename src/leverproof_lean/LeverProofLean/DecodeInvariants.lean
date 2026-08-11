/-
  Decode invariants (I1 / I2 / I6) as a self-contained finite-domain theory.

  Production decode is grammar-constrained end to end. When a *proved*
  complete domain (KERN-02) is a singleton, that symbol is committed with no
  neural ranking. An empty legal domain is a constrained dead end, never a
  full-vocabulary fallback. Rankers may only choose among already-legal
  candidates.

  ``Domain.coverageComplete`` is retained as telemetry / historical artifact
  readability only — it has zero semantic authority (KERN-02 / SLM-517).

  Scope: abstract membership and commit rules. Not a proof that a particular
  PyTorch backend implements them — that remains CI + unit tests
  (`forwards_count == 0`, `verify_decode_invariants`).
-/

import LeverProofLean.ListSet
import LeverProofLean.CompleteDomain

namespace LeverProofLean.DecodeInvariants

open LeverProofLean

abbrev Symbol := Nat

/-- Historical completion domain at one decode step (telemetry carriers). -/
structure Domain where
  candidates : List Symbol
  /-- Telemetry / legacy artifact bit. Zero semantic authority alone (KERN-02). -/
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

/--
  KERN-02 demotion: bare ``coverageComplete`` never yields a singleton token.
  Forced emission requires ``commitComplete`` over a proved ``CompleteDomain``.
-/
def singletonToken? (domain : Domain) : Option Symbol :=
  none

/-- Telemetry helper only — not an authority predicate. -/
def telemetrySuggestsSingleton (domain : Domain) : Bool :=
  domain.coverageComplete && decide (domain.candidates.length = 1)

/-- True singleton under a proved complete domain. -/
def isSingletonComplete (domain : CompleteDomain.CompleteDomain) : Prop :=
  CompleteDomain.isSingleton domain

/--
  Authoritative commit rule over a proved complete domain:

  * unconstrained configs are illegal on the production path;
  * empty live domain ⇒ constrained dead end (never full-vocab fallback);
  * proved singleton ⇒ forced bypass with no ranking;
  * otherwise a ranker may pick only a member of the legal domain.
-/
def commitComplete
    (cfg : Config) (domain : CompleteDomain.CompleteDomain) (rankerTop : Symbol) :
    Commit :=
  if !cfg.grammarConstrained || cfg.allowUnconstrainedFallback then
    .illegal
  else if domain.candidates.isEmpty then
    .deadEnd
  else
    match CompleteDomain.singletonToken? domain with
    | some token => .singletonBypass token
    | none =>
        if rankerTop ∈ domain.candidates then
          .rankedLegal rankerTop
        else
          .illegal

/--
  Legacy commit path over a telemetry ``Domain``.

  Never forces singleton bypass from ``coverageComplete`` (KERN-02). Ranking
  may still select a listed candidate; unknown/partial coverage stays
  non-destructive for empty-domain dead-end handling.
-/
def commit (cfg : Config) (domain : Domain) (rankerTop : Symbol) : Commit :=
  if !cfg.grammarConstrained || cfg.allowUnconstrainedFallback then
    .illegal
  else if domain.candidates.isEmpty then
    .deadEnd
  else if rankerTop ∈ domain.candidates then
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

/-- I2: a proved complete singleton commits by bypass, ignoring the ranker. -/
theorem singleton_bypasses_ranker
    (cfg : Config) (sid : CompleteDomain.StateId) (token rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commitComplete cfg (CompleteDomain.singletonDomain sid token) rankerTop =
      .singletonBypass token := by
  simp [commitComplete, CompleteDomain.singletonToken?, CompleteDomain.singletonDomain,
    hprod.1, hprod.2, List.isEmpty]

/-- Forged coverageComplete alone never forces bypass on the legacy path. -/
theorem forged_coverage_complete_never_bypasses
    (cfg : Config) (token rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commit cfg ⟨[token], true⟩ rankerTop ≠ .singletonBypass token := by
  simp [commit, hprod.1, hprod.2, List.isEmpty]
  intro h
  cases h

/-- I6: empty legal domain is a dead end under production configs. -/
theorem empty_domain_is_dead_end
    (cfg : Config) (coverage : Bool) (rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commit cfg ⟨[], coverage⟩ rankerTop = .deadEnd := by
  simp [commit, hprod.1, hprod.2, List.isEmpty]

/-- I6: empty proved domain is a dead end. -/
theorem empty_complete_domain_is_dead_end
    (cfg : Config) (sid : CompleteDomain.StateId) (rankerTop : Symbol)
    (hprod : productionLegal cfg) :
    commitComplete cfg
      { stateId := sid
        candidates := []
        legal := []
        sound := by intro s hs; cases hs
        complete := by intro s hs; cases hs
        nodup := by simp }
      rankerTop = .deadEnd := by
  simp [commitComplete, hprod.1, hprod.2, List.isEmpty]

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
      · intro h
        injection h with htok
        subst htok
        constructor
        · rfl
        · assumption
      · intro h; cases h

/-- Ranker over a proved domain stays inside candidates. -/
theorem ranked_complete_token_is_legal
    (cfg : Config) (domain : CompleteDomain.CompleteDomain) (rankerTop token : Symbol)
    (h : commitComplete cfg domain rankerTop = .rankedLegal token) :
    token = rankerTop ∧ token ∈ domain.candidates := by
  revert h
  simp only [commitComplete]
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

/-- I1/I2 decision: legacy Domain never surfaces a singleton token (KERN-02). -/
theorem incomplete_coverage_not_singleton (domain : Domain)
    (_h : domain.coverageComplete = false) :
    singletonToken? domain = none := by
  simp [singletonToken?]

/-- Even with coverageComplete=true, legacy Domain yields no singleton token. -/
theorem coverage_complete_flag_not_singleton (domain : Domain)
    (_h : domain.coverageComplete = true) :
    singletonToken? domain = none := by
  simp [singletonToken?]

/-- Bypass commits only emit a domain member (proved-complete path). -/
theorem singleton_bypass_from_token?
    (cfg : Config) (domain : CompleteDomain.CompleteDomain) (rankerTop token : Symbol)
    (hprod : productionLegal cfg)
    (hne : domain.candidates.isEmpty = false)
    (ht : CompleteDomain.singletonToken? domain = some token) :
    commitComplete cfg domain rankerTop = .singletonBypass token := by
  simp [commitComplete, hprod.1, hprod.2, hne, ht]

theorem singleton_token_mem
    (domain : CompleteDomain.CompleteDomain) (token : Symbol)
    (ht : CompleteDomain.singletonToken? domain = some token) :
    token ∈ domain.candidates :=
  CompleteDomain.singleton_token_mem domain token ht

/-- Migration: historical Domain never becomes a proved CompleteDomain by flag. -/
def domainFromLegacyTelemetry (domain : Domain) : CompleteDomain.DomainAuthority :=
  if domain.coverageComplete then
    .partialCoverage
  else
    .unknownCoverage

theorem legacy_telemetry_never_proved (domain : Domain) :
    CompleteDomain.authorizesPruning (domainFromLegacyTelemetry domain) = false := by
  unfold domainFromLegacyTelemetry CompleteDomain.authorizesPruning
  split <;> rfl

end LeverProofLean.DecodeInvariants
