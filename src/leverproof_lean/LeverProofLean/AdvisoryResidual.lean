/-
  Advisory residual plane — non-vacuous claim cores for the harness scorer.

  Models the control flow of `SemanticResidualScorerV1.score` and the
  factor-node membership round-trip used by the SFF suite.

  Not claimed: neural quality, wall-clock, or continuum linear algebra for all
  graphs. Golden matrices are proved/checked exactly and bridged to NumPy tests.
-/

import LeverProofLean.ListSet

namespace LeverProofLean
namespace AdvisoryResidual

open LeverProofLean

abbrev CandId := Nat
abbrev FactorId := Nat

/-! ### Coverage + request (mirrors residual scorer branches) -/

inductive Coverage where
  | complete
  | incomplete
  deriving DecidableEq, Repr, Inhabited

/-- One advisory scoring request over an already-legal candidate set. -/
structure Request where
  coverage : Coverage
  legal : List CandId
  digestMatch : Bool
  legalMatch : Bool
  deriving Repr

/-- Outcome of the advisory control plane (before representation math). -/
inductive Outcome where
  | singletonZeroWork
  | incompleteUnknown
  | staleDigest
  | legalMismatch
  | scored (keys : List CandId) (applications : Nat)
  deriving DecidableEq, Repr

/-- Representation deltas proposed before the legality filter. -/
structure Proposal where
  keys : List CandId
  decodeApply : Bool
  alphaPos : Bool
  deriving Repr

/-- Filter proposal keys to the legal set (harness legality filter). -/
def filterLegal (legal keys : List CandId) : List CandId :=
  keys.filter (fun k => decide (k ∈ legal))

/-- Control-plane score function — branch order matches the Python scorer. -/
def score (req : Request) (prop : Proposal) : Outcome :=
  if req.coverage = .complete ∧ req.legal.length = 1 then
    .singletonZeroWork
  else if req.coverage = .incomplete then
    .incompleteUnknown
  else if !req.digestMatch then
    .staleDigest
  else if !req.legalMatch then
    .legalMismatch
  else
    let keys := filterLegal req.legal prop.keys
    let apps : Nat := if prop.decodeApply ∧ prop.alphaPos then 1 else 0
    .scored keys apps

/-- Complete singleton always takes the zero-work branch (independent of proposal). -/
theorem singleton_is_zero_work
    (legalHead : CandId) (prop : Proposal) (digestMatch legalMatch : Bool) :
    score
      { coverage := .complete, legal := [legalHead], digestMatch := digestMatch,
        legalMatch := legalMatch }
      prop =
      .singletonZeroWork := by
  simp [score]

/-- Incomplete coverage never becomes a scored residual (UNKNOWN preserved). -/
theorem incomplete_not_scored
    (legal : List CandId) (prop : Proposal) (digestMatch legalMatch : Bool) :
    score
      { coverage := .incomplete, legal := legal, digestMatch := digestMatch,
        legalMatch := legalMatch }
      prop =
      .incompleteUnknown := by
  simp [score]

/-- Stale digest rejects before scoring (non-singleton complete domains). -/
theorem stale_digest_rejects_example :
    score
      { coverage := .complete, legal := [0, 1], digestMatch := false, legalMatch := true }
      { keys := [0], decodeApply := true, alphaPos := true } =
      .staleDigest := by
  native_decide

/-- Legal-set mismatch rejects before scoring (non-singleton complete domains). -/
theorem legal_mismatch_rejects_example :
    score
      { coverage := .complete, legal := [0, 1], digestMatch := true, legalMatch := false }
      { keys := [0], decodeApply := true, alphaPos := true } =
      .legalMismatch := by
  native_decide

/-- Executable decode-off check used by golden vectors. -/
def decodeOffZeroApps (legal keys : List CandId) : Bool :=
  match score
    { coverage := .complete, legal := legal, digestMatch := true, legalMatch := true }
    { keys := keys, decodeApply := false, alphaPos := true } with
  | .scored _ apps => apps = 0
  | .singletonZeroWork => true
  | _ => false

theorem decode_off_zero_apps_non_singleton :
    decodeOffZeroApps [0, 1, 2] [0, 9] = true := by
  native_decide

theorem decode_off_zero_apps_singleton :
    decodeOffZeroApps [0] [0, 1] = true := by
  native_decide

/-- Every key surviving filterLegal is a legal candidate. -/
theorem filterLegal_subset (legal keys : List CandId) :
    Subset (filterLegal legal keys) legal := by
  intro k hk
  simp [filterLegal, List.mem_filter, decide_eq_true_eq] at hk
  exact hk.2

/-- Keys outside legal are dropped. -/
theorem filterLegal_drops_illegal
    (legal : List CandId) (k : CandId) (keys : List CandId)
    (h : k ∉ legal) :
    k ∉ filterLegal legal keys := by
  intro hk
  exact h (filterLegal_subset legal keys k hk)

theorem filterLegal_example :
    filterLegal [0, 1, 2] [0, 9, 1, 7] = [0, 1] := by
  native_decide

/-- Scored non-singleton path uses filterLegal keys (executable witness). -/
theorem scored_keys_are_filter_example :
    score
      { coverage := .complete, legal := [0, 1, 2], digestMatch := true, legalMatch := true }
      { keys := [0, 9, 1], decodeApply := true, alphaPos := true } =
      .scored [0, 1] 1 := by
  native_decide

theorem scored_decode_off_zero_apps_example :
    score
      { coverage := .complete, legal := [0, 1, 2], digestMatch := true, legalMatch := true }
      { keys := [0, 9], decodeApply := false, alphaPos := true } =
      .scored [0] 0 := by
  native_decide

/-! ### Factor-node membership round-trip -/

structure FactorMem where
  factorId : FactorId
  nodes : List CandId
  deriving Repr, DecidableEq

def encodeIncidence (factors : List FactorMem) : List (CandId × FactorId) :=
  factors.flatMap fun f => f.nodes.map fun n => (n, f.factorId)

def reconstruct (factorIds : List FactorId) (inc : List (CandId × FactorId)) :
    List FactorMem :=
  factorIds.map fun fid =>
    { factorId := fid
      nodes := (inc.filter (fun p => p.2 = fid)).map (·.1) }

theorem reconstruct_encode_example :
    let factors : List FactorMem := [
      { factorId := 0, nodes := [1, 2] },
      { factorId := 1, nodes := [2, 3] }
    ]
    reconstruct [0, 1] (encodeIncidence factors) = factors := by
  native_decide

theorem reconstruct_encode_singleton :
    reconstruct [7] (encodeIncidence [{ factorId := 7, nodes := [3, 3, 5] }]) =
      [{ factorId := 7, nodes := [3, 3, 5] }] := by
  native_decide

/-! ### Role shuffle preserves membership (models Python `shuffle_port_roles`)

  Ports carry (role, node). Shuffle reassigns roles by left-rotation of the
  role list while **keeping each index's node fixed**. Membership is the node
  list; it is therefore invariant by construction of `shuffleRoles`.
-/

structure Port where
  role : Nat
  node : CandId
  deriving Repr, DecidableEq

def membership (ports : List Port) : List CandId :=
  ports.map (·.node)

/-- Rotate a list left by `k` (mod length). -/
def rotateLeft (xs : List Nat) (k : Nat) : List Nat :=
  match xs with
  | [] => []
  | _ :: _ =>
    let kk := k % xs.length
    xs.drop kk ++ xs.take kk

/-- Reassign roles from `roles'` by index; always keep the original node. -/
def zipRolesOnto (roles' : List Nat) : List Port → List Port
  | [] => []
  | p :: ps =>
    match roles' with
    | [] => { role := p.role, node := p.node } :: zipRolesOnto [] ps
    | r :: rs => { role := r, node := p.node } :: zipRolesOnto rs ps

/-- Role shuffle: rotate role multiset; nodes stay fixed in place. -/
def shuffleRoles (ports : List Port) (k : Nat) : List Port :=
  zipRolesOnto (rotateLeft (ports.map (fun p : Port => p.role)) k) ports

theorem zipRolesOnto_preserves_nodes (roles' : List Nat) (ports : List Port) :
    membership (zipRolesOnto roles' ports) = membership ports := by
  induction ports generalizing roles' with
  | nil => rfl
  | cons p ps ih =>
    cases roles' with
    | nil =>
      simp [zipRolesOnto, membership]
      exact ih []
    | cons r rs =>
      simp [zipRolesOnto, membership]
      exact ih rs

/-- Membership ignores roles; zipRolesOnto keeps `node := p.node`. -/
theorem role_shuffle_preserves_membership (ports : List Port) (k : Nat) :
    membership (shuffleRoles ports k) = membership ports := by
  simpa [shuffleRoles] using zipRolesOnto_preserves_nodes
    (rotateLeft (ports.map (fun p : Port => p.role)) k) ports

/-- Roles move under a non-trivial rotation; nodes do not. -/
theorem role_shuffle_example :
    let ports : List Port := [
      { role := 1, node := 10 },
      { role := 2, node := 20 },
      { role := 3, node := 30 }
    ]
    membership (shuffleRoles ports 1) = [10, 20, 30] ∧
    (shuffleRoles ports 1).map (·.role) = [2, 3, 1] ∧
    membership (shuffleRoles ports 1) = membership ports ∧
    (shuffleRoles ports 1).map (·.role) ≠ ports.map (·.role) := by
  native_decide

/-- Role multiset is preserved by rotation (sorted roles equal). -/
def sortedRoles (ports : List Port) : List Nat :=
  (ports.map (·.role)).mergeSort (· ≤ ·)

theorem role_shuffle_preserves_role_multiset_example :
    let ports : List Port := [
      { role := 1, node := 10 },
      { role := 2, node := 20 },
      { role := 3, node := 30 }
    ]
    sortedRoles (shuffleRoles ports 1) = sortedRoles ports := by
  native_decide
/-! ### Golden incidence degrees (matches NumPy probe B) -/

/-- B columns: factor0 → vertices [1,1,0], factor1 → [1,0,1]. -/
def goldenB : List (List Nat) :=
  [[1, 1, 0], [1, 0, 1]]

def colDegrees (B : List (List Nat)) : List Nat :=
  B.map fun col => col.foldl (· + ·) 0

def rowDegrees (B : List (List Nat)) : List Nat :=
  match B with
  | [] => []
  | c0 :: _ =>
    let n := c0.length
    (List.range n).map fun i =>
      B.foldl (fun acc col => acc + col.getD i 0) 0

theorem golden_degrees :
    colDegrees goldenB = [2, 2] ∧ rowDegrees goldenB = [2, 1, 1] := by
  native_decide

/-- Product of a list of Nats (empty → 1). -/
def listProd (xs : List Nat) : Nat :=
  xs.foldl (· * ·) 1

/-- Product of all entries except index `idx` (empty-safe). -/
def omitProd (xs : List Nat) (idx : Nat) : Nat :=
  listProd ((List.range xs.length).filterMap fun i =>
    if i = idx then none else some (xs.getD i 1))

/--
  Numerator of S_ij * (∏ d_e) * (∏ d_v) under
  S = B D_e^{-1} Bᵀ D_v^{-1} with B stored as list-of-columns.
  S_ij = Σ_k B_ik B_jk / (d_e_k · d_v_j).
-/
def sEntryNum (B : List (List Nat)) (i j : Nat) : Nat :=
  let de := colDegrees B
  let dv := rowDegrees B
  (List.range de.length).foldl (fun acc k =>
    let bik := (B.getD k []).getD i 0
    let bjk := (B.getD k []).getD j 0
    acc + bik * bjk * omitProd de k * omitProd dv j) 0

/-- Column-j mass of S as numerator over (∏ d_e)·(∏ d_v). -/
def sColumnSumNum (B : List (List Nat)) (j : Nat) : Nat :=
  let nRows := (B.headD []).length
  (List.range nRows).foldl (fun acc i => acc + sEntryNum B i j) 0

/-- Target numerator for column-stochastic S (each column sums to 1). -/
def sColumnSumTarget (B : List (List Nat)) : Nat :=
  listProd (colDegrees B) * listProd (rowDegrees B)

/-- Derived from goldenB via the S formula — not a hand-written mass list. -/
def goldenSColumnSumNums : List Nat :=
  (List.range 3).map fun j => sColumnSumNum goldenB j

theorem golden_S_column_masses_from_B :
    let target := sColumnSumTarget goldenB
    goldenSColumnSumNums.all (· = target) = true ∧
    target = 2 * 2 * 2 * 1 * 1 := by
  native_decide

/-- Soft-token non-injectivity (SHIFT correction, finite model). -/
def softToken (weights embed : List Nat) : Nat :=
  (weights.zip embed).foldl (fun acc p => acc + p.1 * p.2) 0

theorem soft_token_collision :
    softToken [1, 0, 0] [7, 7, 3] = softToken [0, 1, 0] [7, 7, 3] ∧
    [1, 0, 0] ≠ [0, 1, 0] := by
  native_decide

/-- Restart contraction factor (1-λ) when λ = num/den ∈ (0,1]. -/
def contractionFactor (lamNum lamDen : Nat) : Nat :=
  lamDen - lamNum

theorem contraction_factor_lt_one
    (lamNum lamDen : Nat) (hpos : 0 < lamNum) (_hle : lamNum ≤ lamDen)
    (hden : 0 < lamDen) :
    contractionFactor lamNum lamDen < lamDen := by
  simpa [contractionFactor] using Nat.sub_lt hden hpos

end AdvisoryResidual
end LeverProofLean
