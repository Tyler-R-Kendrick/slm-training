/-
  Ecosystem-tier partition of formal claim modules.

  Production-core formal modules (Forest / Trace / ExactClosure /
  DecodeInvariants / ListSet) are disjoint from ecosystem-library modules
  (StructuralMetrics and this partition theory). Library growth is modeled as
  a Nat size that never appears in the core formal success predicate.
-/

namespace LeverProofLean.EcosystemTier

/-- Measurement tier for a formal module. -/
inductive Tier where
  | productionCore
  | ecosystemLibrary
  deriving DecidableEq, Repr, BEq

/-- One named formal module and its declared tier. -/
structure FormalModule where
  name : String
  tier : Tier
  deriving Repr, BEq

/-- Production-core formal modules (decode / forest / closure safety). -/
def coreModules : List FormalModule :=
  [ ⟨"ListSet", Tier.productionCore⟩
  , ⟨"Forest", Tier.productionCore⟩
  , ⟨"Trace", Tier.productionCore⟩
  , ⟨"ExactClosure", Tier.productionCore⟩
  , ⟨"DecodeInvariants", Tier.productionCore⟩
  ]

/-- Ecosystem-library formal modules (library-sensitive algebra + partition). -/
def ecosystemModules : List FormalModule :=
  [ ⟨"StructuralMetrics", Tier.ecosystemLibrary⟩
  , ⟨"EcosystemTier", Tier.ecosystemLibrary⟩
  ]

def allModules : List FormalModule :=
  coreModules ++ ecosystemModules

def isCore (m : FormalModule) : Bool :=
  decide (m.tier = Tier.productionCore)

def isEcosystem (m : FormalModule) : Bool :=
  decide (m.tier = Tier.ecosystemLibrary)

/-- Every core module is tagged productionCore. -/
theorem core_modules_are_core :
    coreModules.all isCore = true := by
  decide

/-- Every ecosystem module is tagged ecosystemLibrary. -/
theorem ecosystem_modules_are_ecosystem :
    ecosystemModules.all isEcosystem = true := by
  decide

/-- Core and ecosystem name sets are disjoint (partition). -/
def names (ms : List FormalModule) : List String :=
  ms.map (·.name)

def memString (x : String) : List String → Bool
  | [] => false
  | y :: ys => decide (x = y) || memString x ys

def disjointNames (xs ys : List String) : Bool :=
  xs.all fun x => !(memString x ys)

theorem core_ecosystem_names_disjoint :
    disjointNames (names coreModules) (names ecosystemModules) = true := by
  decide

/--
  Core formal success is a predicate over the core module list only.
  The ecosystem library size parameter is deliberately ignored.
-/
def coreFormalSuccess (coreStatuses : List Bool) (_ecosystemLibrarySize : Nat) : Bool :=
  coreStatuses.length = coreModules.length && coreStatuses.all (· = true)

theorem core_success_ignores_library_size
    (statuses : List Bool) (n m : Nat) :
    coreFormalSuccess statuses n = coreFormalSuccess statuses m := by
  rfl

/-- Growing the library size cannot flip a core success bit by itself. -/
theorem library_growth_preserves_core_success
    (statuses : List Bool) (n k : Nat) :
    coreFormalSuccess statuses n = true →
      coreFormalSuccess statuses (n + k) = true := by
  intro h
  simpa [core_success_ignores_library_size statuses n (n + k)] using h

/-- All-true core statuses of the right length succeed at any library size. -/
theorem all_true_core_succeeds (size : Nat) :
    coreFormalSuccess (coreModules.map fun _ => true) size = true := by
  -- Success ignores library size; reduce to a closed ground goal.
  have h :=
    core_success_ignores_library_size (coreModules.map fun _ => true) size 0
  rw [h]
  decide

end LeverProofLean.EcosystemTier
