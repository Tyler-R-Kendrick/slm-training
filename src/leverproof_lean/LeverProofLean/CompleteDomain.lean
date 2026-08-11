/-
  Proof-bearing finite domains (KERN-02 / SLM-517).

  A complete domain is a finite candidate list bound to a state identity and
  proved equal (as a set) to an explicit legality extension: every candidate is
  Legal, and every Legal member is a candidate. A bare ``coverageComplete``
  Boolean has zero semantic authority and cannot manufacture this structure.
-/

import LeverProofLean.ListSet

namespace LeverProofLean.CompleteDomain

open LeverProofLean

abbrev Symbol := Nat

/-- Opaque state identity binding a domain to one decode / closure state. -/
structure StateId where
  digest : String
  deriving DecidableEq, Repr, BEq

/--
  Proof-bearing complete domain.

  ``sound``: candidate → Legal (every listed candidate is in ``legal``).
  ``complete``: Legal → candidate (every legal member is listed).
  ``nodup``: candidates are a finite set presentation without duplicates.
-/
structure CompleteDomain where
  stateId : StateId
  candidates : List Symbol
  legal : List Symbol
  sound : Subset candidates legal
  complete : Subset legal candidates
  nodup : List.Nodup candidates

/-- Telemetry-only historical Boolean; never authorizes completeness alone. -/
structure LegacyCoverageFlag where
  coverageComplete : Bool
  deriving Repr, BEq

/-- Demotion law: a forged true bit has zero semantic authority. -/
def legacyFlagAuthorizesCompleteness (_ : LegacyCoverageFlag) : Bool :=
  false

theorem forged_coverage_complete_never_authorizes (flag : LegacyCoverageFlag) :
    legacyFlagAuthorizesCompleteness flag = false :=
  rfl

/-- Set equality between candidates and the legality extension. -/
def membersEqLegal (d : CompleteDomain) : Prop :=
  SetEq d.candidates d.legal

theorem proved_members_eq_legal (d : CompleteDomain) : membersEqLegal d :=
  ⟨d.sound, d.complete⟩

def bindsState (d : CompleteDomain) (sid : StateId) : Prop :=
  d.stateId = sid

theorem stale_state_rejects (d : CompleteDomain) (sid : StateId)
    (h : d.stateId ≠ sid) : ¬ bindsState d sid := by
  intro hb
  exact h hb

def isSingleton (d : CompleteDomain) : Prop :=
  d.candidates.length = 1

def singletonToken? (d : CompleteDomain) : Option Symbol :=
  match d.candidates with
  | [token] => some token
  | _ => none

theorem singleton_token_mem (d : CompleteDomain) (token : Symbol)
    (ht : singletonToken? d = some token) :
    token ∈ d.candidates := by
  unfold singletonToken? at ht
  match hcands : d.candidates with
  | [] => simp [hcands] at ht
  | t :: rest =>
      match rest with
      | [] =>
          simp [hcands] at ht
          subst t
          simp
      | _ :: _ => simp [hcands] at ht

theorem singleton_is_legal (d : CompleteDomain) (token : Symbol)
    (ht : singletonToken? d = some token) :
    token ∈ d.legal :=
  d.sound token (singleton_token_mem d token ht)

theorem singleton_unique_legal (d : CompleteDomain) (token other : Symbol)
    (ht : singletonToken? d = some token)
    (ho : other ∈ d.legal) :
    other = token := by
  have hoc := d.complete other ho
  have hmem := singleton_token_mem d token ht
  unfold singletonToken? at ht
  match hcands : d.candidates with
  | [t] =>
      simp [hcands] at ht hmem hoc
      subst t
      exact hoc
  | [] => simp [hcands] at ht
  | _ :: _ :: _ => simp [hcands] at ht

/-- Partial / unknown domains remain non-authoritative. -/
inductive DomainAuthority where
  | provedComplete (d : CompleteDomain)
  | partialCoverage
  | unknownCoverage

def authorizesPruning : DomainAuthority → Bool
  | .provedComplete _ => true
  | .partialCoverage => false
  | .unknownCoverage => false

def authorizesForcedEmission : DomainAuthority → Bool
  | .provedComplete d =>
      match singletonToken? d with
      | some _ => true
      | none => false
  | .partialCoverage => false
  | .unknownCoverage => false

theorem partial_never_authorizes_pruning :
    authorizesPruning .partialCoverage = false :=
  rfl

theorem unknown_never_authorizes_pruning :
    authorizesPruning .unknownCoverage = false :=
  rfl

theorem legacy_flag_never_yields_proved
    (flag : LegacyCoverageFlag) :
    legacyFlagAuthorizesCompleteness flag = false ∧
      authorizesPruning .partialCoverage = false :=
  ⟨rfl, rfl⟩

/-- Construct a singleton complete domain when legality is exactly ``[token]``. -/
def singletonDomain (sid : StateId) (token : Symbol) : CompleteDomain :=
  { stateId := sid
    candidates := [token]
    legal := [token]
    sound := by
      intro s hs
      simpa using hs
    complete := by
      intro s hs
      simpa using hs
    nodup := by simp }

theorem singleton_domain_forces_bypass_token (sid : StateId) (token : Symbol) :
    singletonToken? (singletonDomain sid token) = some token := by
  simp [singletonToken?, singletonDomain]

end LeverProofLean.CompleteDomain
