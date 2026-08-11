/-
  Finite completion-search upper bounds (KERN-03 / SLM-521).

  Abstract finite-search model parameterized by hole/domain sizes (from
  proof-bearing ``CompleteDomain`` lengths) and per-expansion / per-verification
  *query* costs. Bounds are stated against the named
  ``ExpansionVerificationQueryModel`` — never as wall-clock latency claims.

  Preconditions (explicit on every theorem that needs them):
  * domain completeness / finite sizes from proved domains;
  * algorithm correspondence: exhaustive prefix-tree enumeration over those sizes;
  * cost model: query counts under ``ExpansionVerificationQueryModel``.
-/

import LeverProofLean.CompleteDomain

namespace LeverProofLean.FiniteSearchBounds

open LeverProofLean

/--
  Named machine/query model.

  Counts expansion queries (prefix-tree nodes visited) and verification queries
  (complete-assignment checks). This is **not** a wall-clock cost model.
-/
structure ExpansionVerificationQueryModel where
  costPerExpansion : Nat
  costPerVerification : Nat
  deriving Repr, BEq, DecidableEq

/-- Ordered hole domain sizes ``d_i`` plus the named query-cost model. -/
structure FiniteSearchModel where
  domainSizes : List Nat
  costModel : ExpansionVerificationQueryModel
  deriving Repr, BEq

def holeCount (m : FiniteSearchModel) : Nat :=
  m.domainSizes.length

/-- Empty product is 1. -/
def listProd : List Nat → Nat
  | [] => 1
  | d :: ds => d * listProd ds

def listSum : List Nat → Nat
  | [] => 0
  | x :: xs => x + listSum xs

/-- Complete-assignment count ``P = ∏ d_i``. -/
def completeAssignmentCount (sizes : List Nat) : Nat :=
  listProd sizes

/--
  Cumulative products ``∏_{i ≤ k} d_i`` for each hole index ``k``
  (0-based, left-to-right). Empty sizes → empty list.
-/
def prefixProductsAux (pref : Nat) : List Nat → List Nat
  | [] => []
  | d :: ds =>
      let p := pref * d
      p :: prefixProductsAux p ds

def prefixProducts (sizes : List Nat) : List Nat :=
  prefixProductsAux 1 sizes

/-- Prefix-tree node bound ``1 + Σ_k ∏_{i ≤ k} d_i`` (root plus every depth). -/
def prefixTreeNodeBound (sizes : List Nat) : Nat :=
  1 + listSum (prefixProducts sizes)

/-- Coarse bound ``1 + h·P`` when every domain size is at least 1. -/
def coarseNodeBound (sizes : List Nat) : Nat :=
  1 + sizes.length * completeAssignmentCount sizes

def domainSizeOf (d : CompleteDomain.CompleteDomain) : Nat :=
  d.candidates.length

def sizesFromProvedDomains (ds : List CompleteDomain.CompleteDomain) : List Nat :=
  ds.map domainSizeOf

def modelFromProvedDomains
    (ds : List CompleteDomain.CompleteDomain)
    (cost : ExpansionVerificationQueryModel) : FiniteSearchModel :=
  { domainSizes := sizesFromProvedDomains ds, costModel := cost }

def expansionQueryBound (m : FiniteSearchModel) : Nat :=
  prefixTreeNodeBound m.domainSizes * m.costModel.costPerExpansion

def verificationQueryBound (m : FiniteSearchModel) : Nat :=
  completeAssignmentCount m.domainSizes * m.costModel.costPerVerification

/-- Total query-cost upper bound under the named model (not wall-clock). -/
def totalQueryBound (m : FiniteSearchModel) : Nat :=
  expansionQueryBound m + verificationQueryBound m

/-! ### Basic product / sum facts -/

theorem listProd_nil : listProd [] = 1 := rfl

theorem listProd_cons (d : Nat) (ds : List Nat) :
    listProd (d :: ds) = d * listProd ds :=
  rfl

theorem completeAssignmentCount_eq (sizes : List Nat) :
    completeAssignmentCount sizes = listProd sizes :=
  rfl

theorem prefixTreeNodeBound_nil :
    prefixTreeNodeBound [] = 1 := by
  simp [prefixTreeNodeBound, prefixProducts, prefixProductsAux, listSum]

theorem completeAssignmentCount_nil :
    completeAssignmentCount [] = 1 :=
  rfl

theorem completeAssignmentCount_zero_head (ds : List Nat) :
    completeAssignmentCount (0 :: ds) = 0 := by
  simp [completeAssignmentCount, listProd]

theorem prefix_products_aux_mul (pref d : Nat) (ds : List Nat) :
    prefixProductsAux pref (d :: ds) =
      (pref * d) :: prefixProductsAux (pref * d) ds :=
  rfl

theorem listSum_map_mul_le (c : Nat) (xs : List Nat)
    (h : ∀ x ∈ xs, x ≤ c) :
    listSum xs ≤ xs.length * c := by
  induction xs with
  | nil => simp [listSum]
  | cons x xs ih =>
      have hx : x ≤ c := h x (by simp)
      have hxs : ∀ y ∈ xs, y ≤ c := fun y hy => h y (by simp [hy])
      have ih' := ih hxs
      calc
        x + listSum xs ≤ c + listSum xs := Nat.add_le_add_right hx _
        _ ≤ c + xs.length * c := Nat.add_le_add_left ih' _
        _ = (xs.length + 1) * c := by
            rw [Nat.succ_mul]
            ac_rfl

/-- Every domain size ≥ 1 implies the running product is ≥ 1. -/
theorem listProd_pos_of_pos :
    ∀ (sizes : List Nat),
      (∀ d ∈ sizes, 1 ≤ d) → 1 ≤ listProd sizes
  | [], _ => Nat.le_refl 1
  | d :: ds, h => by
      have hd : 1 ≤ d := h d (by simp)
      have hds : ∀ x ∈ ds, 1 ≤ x := fun x hx => h x (by simp [hx])
      have ih := listProd_pos_of_pos ds hds
      have : 1 * 1 ≤ d * listProd ds := Nat.mul_le_mul hd ih
      simpa [listProd] using this

/-- Extending a positive suffix cannot shrink the product. -/
theorem listProd_le_append_of_pos (pref rest : List Nat)
    (hrest : ∀ d ∈ rest, 1 ≤ d) :
    listProd pref ≤ listProd (pref ++ rest) := by
  induction rest generalizing pref with
  | nil => simp
  | cons d ds ih =>
      have hd : 1 ≤ d := hrest d (by simp)
      have hds : ∀ x ∈ ds, 1 ≤ x := fun x hx => hrest x (by simp [hx])
      -- listProd (pref ++ d :: ds) = listProd ((pref ++ [d]) ++ ds)
      have step : listProd pref ≤ listProd (pref ++ [d]) := by
        -- listProd (pref ++ [d]) = listProd pref * d
        induction pref with
        | nil =>
            simpa [listProd] using hd
        | cons p ps ihp =>
            simp [listProd] at *
            exact Nat.mul_le_mul_left p ihp
      have ih' := ih (pref ++ [d]) hds
      calc
        listProd pref ≤ listProd (pref ++ [d]) := step
        _ ≤ listProd ((pref ++ [d]) ++ ds) := ih'
        _ = listProd (pref ++ (d :: ds)) := by simp

/-- Each prefix product is ≤ ``P`` when every ``d_i ≥ 1``. -/
theorem prefix_products_le_P (sizes : List Nat)
    (hpos : ∀ d ∈ sizes, 1 ≤ d) :
    ∀ p ∈ prefixProducts sizes, p ≤ completeAssignmentCount sizes := by
  -- Prove a stronger statement about the aux accumulator.
  have aux :
      ∀ (pref : Nat) (rest : List Nat),
        (∀ d ∈ rest, 1 ≤ d) →
        pref ≤ pref * listProd rest →
        ∀ p ∈ prefixProductsAux pref rest,
          p ≤ pref * listProd rest := by
    intro pref rest
    induction rest generalizing pref with
    | nil =>
        intro _ _ p hp
        cases hp
    | cons d ds ih =>
        intro hrest hpref p hp
        have hd : 1 ≤ d := hrest d (by simp)
        have hds : ∀ x ∈ ds, 1 ≤ x := fun x hx => hrest x (by simp [hx])
        simp [prefixProductsAux] at hp
        cases hp with
        | inl heq =>
            -- p = pref * d
            subst heq
            have hrest_prod : 1 ≤ listProd ds := listProd_pos_of_pos ds hds
            have : pref * d ≤ pref * d * listProd ds := by
              have := Nat.mul_le_mul_left (pref * d) hrest_prod
              simpa [Nat.mul_one] using this
            simpa [listProd, Nat.mul_assoc] using this
        | inr hmem =>
            have hpref' : pref * d ≤ (pref * d) * listProd ds := by
              have hrest_prod : 1 ≤ listProd ds := listProd_pos_of_pos ds hds
              have := Nat.mul_le_mul_left (pref * d) hrest_prod
              simpa [Nat.mul_one] using this
            have ih' := ih (pref * d) hds hpref'
            have := ih' p hmem
            simpa [listProd, Nat.mul_assoc] using this
  intro p hp
  have hpref0 : (1 : Nat) ≤ 1 * listProd sizes := by
    simpa using listProd_pos_of_pos sizes hpos
  simpa [prefixProducts, completeAssignmentCount, Nat.one_mul] using
    aux 1 sizes hpos hpref0 p hp

/--
  Derived coarse bound: when every domain size is ≥ 1,
  ``prefixTreeNodeBound ≤ 1 + h·P``.
-/
theorem prefix_tree_le_one_plus_holes_mul_assignments (sizes : List Nat)
    (hpos : ∀ d ∈ sizes, 1 ≤ d) :
    prefixTreeNodeBound sizes ≤ coarseNodeBound sizes := by
  unfold prefixTreeNodeBound coarseNodeBound
  have hsum :
      listSum (prefixProducts sizes) ≤
        (prefixProducts sizes).length * completeAssignmentCount sizes :=
    listSum_map_mul_le (completeAssignmentCount sizes) (prefixProducts sizes)
      (prefix_products_le_P sizes hpos)
  have hlen : (prefixProducts sizes).length = sizes.length := by
    -- prefixProductsAux preserves length
    have aux_len : ∀ pref rest, (prefixProductsAux pref rest).length = rest.length := by
      intro pref rest
      induction rest generalizing pref with
      | nil => simp [prefixProductsAux]
      | cons d ds ih => simp [prefixProductsAux, ih]
    simpa [prefixProducts] using aux_len 1 sizes
  rw [hlen] at hsum
  exact Nat.add_le_add_left hsum 1

/-! ### Pruning / memoization cannot increase the declared upper bound -/

/--
  A search trace under the enumeration algorithm: expansion visits and
  verification queries. Correspondence assumption: expansions enumerate a
  subset of the prefix tree; verifications enumerate a subset of complete
  assignments.
-/
structure SearchTrace where
  expansions : Nat
  verifications : Nat
  deriving Repr, BEq

def respectsUpperBounds (sizes : List Nat) (t : SearchTrace) : Prop :=
  t.expansions ≤ prefixTreeNodeBound sizes ∧
    t.verifications ≤ completeAssignmentCount sizes

/--
  Pruning or memoization that only *reduces* visits cannot raise the
  certified upper bound (under the subset-visit correspondence assumption).
-/
theorem pruning_memo_cannot_increase_bound
    (sizes : List Nat) (t t' : SearchTrace)
    (h : respectsUpperBounds sizes t)
    (hle :
      t'.expansions ≤ t.expansions ∧
        t'.verifications ≤ t.verifications) :
    respectsUpperBounds sizes t' := by
  refine ⟨Nat.le_trans hle.1 h.1, Nat.le_trans hle.2 h.2⟩

/-- Unit-cost query model used by fixture cross-checks. -/
def unitCostModel : ExpansionVerificationQueryModel :=
  { costPerExpansion := 1, costPerVerification := 1 }

/-- Concrete fixture: domains [2, 3] → P = 6, prefix nodes = 1+2+6 = 9. -/
theorem fixture_two_three_assignment_count :
    completeAssignmentCount [2, 3] = 6 := by
  decide

theorem fixture_two_three_prefix_bound :
    prefixTreeNodeBound [2, 3] = 9 := by
  decide

theorem fixture_two_three_coarse :
    prefixTreeNodeBound [2, 3] ≤ coarseNodeBound [2, 3] := by
  decide

theorem fixture_empty_holes :
    completeAssignmentCount ([] : List Nat) = 1 ∧
      prefixTreeNodeBound [] = 1 := by
  decide

theorem fixture_zero_domain :
    completeAssignmentCount [2, 0, 4] = 0 ∧
      prefixTreeNodeBound [2, 0, 4] = 1 + 2 + 0 + 0 := by
  decide

theorem fixture_singleton_domains_from_proved :
    let sid : CompleteDomain.StateId := ⟨"s1"⟩
    let d0 := CompleteDomain.singletonDomain sid 7
    let d1 := CompleteDomain.singletonDomain sid 8
    completeAssignmentCount (sizesFromProvedDomains [d0, d1]) = 1 ∧
      prefixTreeNodeBound (sizesFromProvedDomains [d0, d1]) = 3 := by
  decide

end LeverProofLean.FiniteSearchBounds
