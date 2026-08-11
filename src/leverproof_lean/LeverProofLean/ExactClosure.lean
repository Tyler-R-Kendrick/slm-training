/-
  VSS exact-closure theory (certificate-checked domain reduction).

  Axiomatizes the VSS1-01 fixed-point operator as a pure membership model:
  * only replay-valid UNSUPPORTED candidates may be removed;
  * SUPPORTED / UNKNOWN / failed-replay never shrink a domain;
  * one pass is monotone (live set only shrinks);
  * the multi-pass fixpoint is a monotone iteration of passes;
  * certified bottom means every value of some hole was certificate-removed.

  Soft scores never appear. This is the formal safety layer for exact closure;
  it does not prove oracle completeness or empirical solver quality.
-/

import LeverProofLean.ListSet
import LeverProofLean.CompleteDomain

namespace LeverProofLean.ExactClosure

open LeverProofLean

abbrev Candidate := Nat
abbrev HoleId := Nat

/-- Local sum/prod (KERN-05); mirrors FiniteSearchBounds without importing it. -/
def listSum : List Nat → Nat
  | [] => 0
  | x :: xs => x + listSum xs

def listProd : List Nat → Nat
  | [] => 1
  | d :: ds => d * listProd ds

def completeAssignmentCount (sizes : List Nat) : Nat :=
  listProd sizes

/-- Support oracle verdict (VSS0 tri-state). -/
inductive Verdict where
  | supported
  | unsupported
  | unknown
  deriving DecidableEq, Repr, BEq

/-- One oracle answer for a single (hole, value) query. -/
structure QueryResult where
  hole : HoleId
  candidate : Candidate
  verdict : Verdict
  /-- Replay succeeded against the pass-start state. Only meaningful for unsupported. -/
  replayOk : Bool
  deriving Repr, BEq

/-- Destructive removal is allowed only for replay-valid UNSUPPORTED. -/
def removable (r : QueryResult) : Bool :=
  match r.verdict with
  | .unsupported => r.replayOk
  | _ => false

/-- Whether a candidate is removable according to a batch of results. -/
def isRemovable (results : List QueryResult) (c : Candidate) : Bool :=
  results.any fun r => decide (r.candidate = c) && removable r

/--
  One exact-closure pass over a live domain: drop only certificate-backed
  unsupported candidates. Pass-consistency is modeled by treating `results`
  as computed against a single frozen pre-pass state.
-/
def closePass (live : List Candidate) (results : List QueryResult) : List Candidate :=
  live.filter fun c => !(isRemovable results c)

theorem closePass_subset (live : List Candidate) (results : List QueryResult) :
    Subset (closePass live results) live := by
  intro c hc
  exact (List.mem_filter.mp hc).1

theorem closePass_never_adds (live : List Candidate) (results : List QueryResult)
    (c : Candidate) (hc : c ∈ closePass live results) : c ∈ live :=
  closePass_subset live results c hc

/-- SUPPORTED answers never authorize removal. -/
theorem supported_not_removable (r : QueryResult)
    (h : r.verdict = Verdict.supported) : removable r = false := by
  simp [removable, h]

/-- UNKNOWN answers never authorize removal. -/
theorem unknown_not_removable (r : QueryResult)
    (h : r.verdict = Verdict.unknown) : removable r = false := by
  simp [removable, h]

/-- Failed certificate replay never authorizes removal. -/
theorem failed_replay_not_removable (r : QueryResult)
    (h : r.replayOk = false) : removable r = false := by
  cases hv : r.verdict <;> simp [removable, hv, h]

/-- A candidate that no result marks removable survives the pass. -/
theorem survivor_if_not_removable
    (live : List Candidate) (results : List QueryResult) (c : Candidate)
    (hin : c ∈ live) (hnot : isRemovable results c = false) :
    c ∈ closePass live results := by
  simp only [closePass, List.mem_filter, hin, hnot, Bool.not_false, and_self]

/-- A candidate marked removable leaves the live set. -/
theorem removable_leaves
    (live : List Candidate) (results : List QueryResult) (c : Candidate)
    (hrem : isRemovable results c = true) :
    c ∉ closePass live results := by
  intro hc
  have := (List.mem_filter.mp hc).2
  simp [hrem] at this

/-- Multi-pass iteration: fold closePass over a list of result batches. -/
def iterate (live : List Candidate) : List (List QueryResult) → List Candidate
  | [] => live
  | batch :: rest => iterate (closePass live batch) rest

theorem iterate_subset (live : List Candidate) (batches : List (List QueryResult)) :
    Subset (iterate live batches) live := by
  induction batches generalizing live with
  | nil =>
      intro c hc
      exact hc
  | cons batch rest ih =>
      intro c hc
      have hmid := ih (closePass live batch) c hc
      exact closePass_subset live batch c hmid

/-- Fixed point: a pass that removes nothing leaves the live set unchanged. -/
def isFixedPoint (live : List Candidate) (results : List QueryResult) : Prop :=
  SetEq (closePass live results) live

theorem fixed_point_keeps_all
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ c ∈ live, isRemovable results c = false) :
    isFixedPoint live results := by
  constructor
  · exact closePass_subset live results
  · intro c hc
    exact survivor_if_not_removable live results c hc (h c hc)

/--
  Certified bottom for a single hole: every value of that hole was removed under
  an accepted certificate. Modeled as "no live survivors remain".
-/
def certifiedBottom (live : List Candidate) : Prop :=
  live = []

private theorem filter_eq_nil_of_all_false
    (xs : List Candidate) (p : Candidate → Bool)
    (h : ∀ c ∈ xs, p c = false) :
    xs.filter p = [] := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
      have hhead := h head (by simp)
      have htail : ∀ c ∈ tail, p c = false := by
        intro c hc
        exact h c (by simp [hc])
      simp [List.filter, hhead, ih htail]

theorem certified_bottom_of_all_removed
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ c ∈ live, isRemovable results c = true) :
    certifiedBottom (closePass live results) := by
  unfold certifiedBottom closePass
  -- filter keeps when !isRemovable; all isRemovable=true ⇒ keep predicate is false
  apply filter_eq_nil_of_all_false
  intro c hc
  simp [h c hc]

/-- Empty result list removes nothing. -/
theorem empty_results_fixed_point (live : List Candidate) :
    isFixedPoint live [] := by
  apply fixed_point_keeps_all
  intro c _hc
  simp [isRemovable]

/-- A single non-removable result cannot shrink the domain. -/
theorem nonremovable_result_fixed_point
    (live : List Candidate) (r : QueryResult)
    (hr : removable r = false) :
    isFixedPoint live [r] := by
  apply fixed_point_keeps_all
  intro c _hc
  simp [isRemovable, hr]

/--
  Honesty composition: if every individual result is non-removable, the batch
  is a fixed point. Proved by induction on the result list.
-/
theorem honest_when_no_accepted_unsupported
    (live : List Candidate) (results : List QueryResult)
    (h : ∀ r ∈ results, removable r = false) :
    isFixedPoint live results := by
  apply fixed_point_keeps_all
  intro c _hc
  induction results with
  | nil =>
      simp [isRemovable]
  | cons r rest ih =>
      have hr : removable r = false := h r (by simp)
      have hrest : ∀ r' ∈ rest, removable r' = false := by
        intro r' hr'
        exact h r' (List.mem_cons_of_mem r hr')
      have ih' := ih hrest
      -- isRemovable (r :: rest) c = ((r.candidate == c) && removable r) || isRemovable rest c
      simp only [isRemovable, List.any_cons, hr, Bool.and_false, Bool.false_or]
      exact ih'

/-! ### KERN-02 bridge: finite search / singleton require proved complete domains -/

/--
  Finite completion search is authorized only over a proved complete domain
  bound to the query state. A bare coverage Boolean is not accepted here.
-/
def finiteSearchAuthorized (auth : CompleteDomain.DomainAuthority) (sid : CompleteDomain.StateId) : Bool :=
  match auth with
  | .provedComplete d => decide (d.stateId = sid)
  | .partialCoverage => false
  | .unknownCoverage => false

theorem finite_search_rejects_partial (sid : CompleteDomain.StateId) :
    finiteSearchAuthorized .partialCoverage sid = false := by
  simp [finiteSearchAuthorized]

theorem finite_search_rejects_unknown (sid : CompleteDomain.StateId) :
    finiteSearchAuthorized .unknownCoverage sid = false := by
  simp [finiteSearchAuthorized]

theorem finite_search_rejects_stale_state
    (d : CompleteDomain.CompleteDomain) (sid : CompleteDomain.StateId) (h : d.stateId ≠ sid) :
    finiteSearchAuthorized (.provedComplete d) sid = false := by
  simp [finiteSearchAuthorized, h]

/-- Certified bottom over a proved-empty complete domain. -/
theorem certified_bottom_of_proved_empty
    (d : CompleteDomain.CompleteDomain) (hempty : d.candidates = []) :
    certifiedBottom d.candidates := by
  simp [certifiedBottom, hempty]

/--
  Singleton survivor conclusions require a proved complete domain. Legacy
  coverage flags never authorize the conclusion on their own.
-/
theorem singleton_survivor_requires_proved_domain
    (flag : CompleteDomain.LegacyCoverageFlag) :
    CompleteDomain.legacyFlagAuthorizesCompleteness flag = false :=
  CompleteDomain.forged_coverage_complete_never_authorizes flag

/-! ### KERN-05: strict progress and finite stabilization bounds

  Monotonicity (`closePass_subset`) is strengthened to a dichotomy: every pass
  is either a membership fixed point or removes at least one previously live
  candidate. Finite stabilization then follows from the height of the product
  of finite subset lattices (one chain per hole): at most ``Σ d_i`` strict
  candidate removals, or ``Σ (d_i - 1)`` when every hole is required to stay
  nonempty. Closure-pass complexity (``Σ d_i``) is separated from full
  assignment-search complexity (``∏ d_i``). Batch fairness/coverage is stated
  explicitly; maximality / oracle-completeness of a fixed point is **not**
  claimed without those assumptions.
-/

/-- Candidates dropped by one pass (certificate-backed removable filter). -/
def removed (live : List Candidate) (results : List QueryResult) : List Candidate :=
  live.filter fun c => isRemovable results c

theorem mem_removed_iff
    (live : List Candidate) (results : List QueryResult) (c : Candidate) :
    c ∈ removed live results ↔ c ∈ live ∧ isRemovable results c = true := by
  simp [removed, List.mem_filter]

/-- Batch names a candidate (coverage / fairness atom). -/
def batchMentions (results : List QueryResult) (c : Candidate) : Bool :=
  results.any fun r => decide (r.candidate = c)

/--
  Fairness/coverage precondition for charging progress against a batch: every
  live candidate the batch marks removable is named in the batch. Under the
  current ``isRemovable`` encoding this holds definitionally; it is kept as an
  explicit obligation so alternate batching encodings cannot silently drop it.
  This does **not** prove oracle completeness or maximality of a fixed point.
-/
def batchCoversRemovable (live : List Candidate) (results : List QueryResult) : Prop :=
  ∀ c ∈ live, isRemovable results c = true → batchMentions results c = true

theorem removable_implies_batch_mentions
    (results : List QueryResult) (c : Candidate)
    (h : isRemovable results c = true) :
    batchMentions results c = true := by
  -- isRemovable requires some result with matching candidate; that implies batchMentions.
  simp only [isRemovable, batchMentions, List.any_eq_true, Bool.and_eq_true,
    decide_eq_true_eq] at h ⊢
  obtain ⟨r, hr, ⟨heq, _⟩⟩ := h
  exact ⟨r, hr, heq⟩

theorem batchCoversRemovable_holds
    (live : List Candidate) (results : List QueryResult) :
    batchCoversRemovable live results := by
  intro c _ hc
  exact removable_implies_batch_mentions results c hc

private theorem length_filter_partition
    (xs : List Candidate) (p : Candidate → Bool) :
    (xs.filter p).length + (xs.filter fun x => !(p x)).length = xs.length := by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      cases hx : p x
      · -- p x = false
        simp [List.filter, hx]
        -- goal: (filter p xs).length + ((filter !p xs).length + 1) = xs.length + 1
        have := ih
        omega
      · -- p x = true
        simp [List.filter, hx]
        have := ih
        omega

theorem closePass_length_add_removed
    (live : List Candidate) (results : List QueryResult) :
    (closePass live results).length + (removed live results).length = live.length := by
  have h := length_filter_partition live fun c => isRemovable results c
  -- |rem| + |!rem| = |live|; closePass = filter !rem
  simpa [closePass, removed, Nat.add_comm] using h

theorem closePass_length_le
    (live : List Candidate) (results : List QueryResult) :
    (closePass live results).length ≤ live.length := by
  have h := closePass_length_add_removed live results
  omega

/-- Membership form of strict progress: not a fixed point ↔ some live candidate leaves. -/
theorem not_fixed_point_iff_exists_removal
    (live : List Candidate) (results : List QueryResult) :
    ¬ isFixedPoint live results ↔
      ∃ c, c ∈ live ∧ c ∉ closePass live results := by
  constructor
  · intro h
    have hsub := closePass_subset live results
    have hnot : ¬ Subset live (closePass live results) := fun hs => h ⟨hsub, hs⟩
    have hex : ∃ c, ¬ (c ∈ live → c ∈ closePass live results) :=
      Classical.not_forall.mp hnot
    obtain ⟨c, hc⟩ := hex
    have hc' := not_imp.mp hc
    exact ⟨c, hc'.1, hc'.2⟩
  · intro ⟨c, hin, hout⟩ hfp
    exact hout (hfp.2 c hin)

/-- Dichotomy: every pass is a fixed point or removes a previously live candidate. -/
theorem strict_progress_or_fixed_point
    (live : List Candidate) (results : List QueryResult) :
    isFixedPoint live results ∨
      ∃ c, c ∈ live ∧ c ∉ closePass live results := by
  by_cases h : isFixedPoint live results
  · exact Or.inl h
  · exact Or.inr ((not_fixed_point_iff_exists_removal live results).mp h)

private theorem removable_of_live_not_survivor
    (live : List Candidate) (results : List QueryResult) (c : Candidate)
    (hin : c ∈ live) (hout : c ∉ closePass live results) :
    isRemovable results c = true := by
  by_cases hr : isRemovable results c = true
  · exact hr
  · have hfalse : isRemovable results c = false := Bool.eq_false_iff.mpr hr
    exact (hout (survivor_if_not_removable live results c hin hfalse)).elim

/-- Under uniqueness (Nodup) and batch coverage, a non-fixed pass decreases length. -/
theorem strict_progress_decreases_length
    (live : List Candidate) (results : List QueryResult)
    (_hnodup : live.Nodup)
    (hcov : batchCoversRemovable live results)
    (h : ¬ isFixedPoint live results) :
    (closePass live results).length < live.length := by
  have ⟨c, hin, hout⟩ := (not_fixed_point_iff_exists_removal live results).mp h
  have hrem := removable_of_live_not_survivor live results c hin hout
  have _ := hcov c hin hrem
  have hin_rem : c ∈ removed live results :=
    (mem_removed_iff live results c).mpr ⟨hin, hrem⟩
  have hpos : 0 < (removed live results).length :=
    List.length_pos_of_mem hin_rem
  have hsum := closePass_length_add_removed live results
  omega

/-- Flat-list form: non-fixed ⇒ removed list nonempty. -/
theorem not_fixed_implies_removed_nonempty
    (live : List Candidate) (results : List QueryResult)
    (h : ¬ isFixedPoint live results) :
    removed live results ≠ [] := by
  obtain ⟨c, hin, hout⟩ := (not_fixed_point_iff_exists_removal live results).mp h
  exact List.ne_nil_of_mem
    ((mem_removed_iff live results c).mpr
      ⟨hin, removable_of_live_not_survivor live results c hin hout⟩)

theorem not_fixed_implies_length_decreases
    (live : List Candidate) (results : List QueryResult)
    (h : ¬ isFixedPoint live results) :
    (closePass live results).length < live.length := by
  have hne := not_fixed_implies_removed_nonempty live results h
  have hpos : 0 < (removed live results).length := by
    cases hrem : removed live results with
    | nil => exact (hne hrem).elim
    | cons _ _ => exact Nat.zero_lt_succ _
  have hsum := closePass_length_add_removed live results
  omega

/-- Candidates removed by one pass (cardinality). -/
def removedCount (live : List Candidate) (results : List QueryResult) : Nat :=
  live.length - (closePass live results).length

theorem removedCount_eq_removed_length
    (live : List Candidate) (results : List QueryResult) :
    removedCount live results = (removed live results).length := by
  have h := closePass_length_add_removed live results
  unfold removedCount
  omega

/-- Total strict candidate removals across an iteration of batches. -/
def totalRemovedAcross (live : List Candidate) : List (List QueryResult) → Nat
  | [] => 0
  | batch :: rest =>
      removedCount live batch +
        totalRemovedAcross (closePass live batch) rest

/-- Finite stabilization (flat live bag): removals ≤ initial live cardinality. -/
theorem totalRemovedAcross_le_live
    (live : List Candidate) (batches : List (List QueryResult)) :
    totalRemovedAcross live batches ≤ live.length := by
  induction batches generalizing live with
  | nil => simp [totalRemovedAcross]
  | cons batch rest ih =>
      simp only [totalRemovedAcross, removedCount]
      have hle : (closePass live batch).length ≤ live.length :=
        closePass_length_le live batch
      have ih' : totalRemovedAcross (closePass live batch) rest ≤
          (closePass live batch).length := ih (closePass live batch)
      calc
        live.length - (closePass live batch).length +
            totalRemovedAcross (closePass live batch) rest
          ≤ live.length - (closePass live batch).length +
              (closePass live batch).length := Nat.add_le_add_left ih' _
        _ = live.length := Nat.sub_add_cancel hle

/-! #### Product-lattice height (multi-hole domains) -/

/-- Per-hole live domains (ordered holes). -/
abbrev HoleDomains := List (List Candidate)

def holeSizes (holes : HoleDomains) : List Nat :=
  holes.map List.length

def totalLiveCount (holes : HoleDomains) : Nat :=
  listSum (holeSizes holes)

def closePassHoles (holes : HoleDomains) (results : List QueryResult) : HoleDomains :=
  holes.map fun live => closePass live results

def allNonempty (holes : HoleDomains) : Prop :=
  ∀ h ∈ holes, h ≠ []

/-- Height of the product of finite subset lattices: ``Σ d_i``. -/
def strictRemovalUpperBound (domainSizes : List Nat) : Nat :=
  listSum domainSizes

/-- Nonempty-invariant height: ``Σ (d_i - 1)`` (Nat saturating subtraction). -/
def nonemptyInvariantUpperBound (domainSizes : List Nat) : Nat :=
  listSum (domainSizes.map fun d => d - 1)

/-- Assignment-search complexity measure (KERN-03): ``∏ d_i``. -/
def assignmentSearchBound (domainSizes : List Nat) : Nat :=
  completeAssignmentCount domainSizes

theorem strictRemovalUpperBound_eq_sum (sizes : List Nat) :
    strictRemovalUpperBound sizes = listSum sizes :=
  rfl

theorem nonemptyInvariantUpperBound_eq (sizes : List Nat) :
    nonemptyInvariantUpperBound sizes =
      listSum (sizes.map fun d => d - 1) :=
  rfl

theorem assignmentSearchBound_eq_prod (sizes : List Nat) :
    assignmentSearchBound sizes = listProd sizes :=
  rfl

/-- Witness that closure-pass complexity (``Σ``) ≠ assignment-search (``∏``). -/
theorem closure_complexity_separated_from_assignment_search :
    strictRemovalUpperBound [1, 1, 1] = 3 ∧
      assignmentSearchBound [1, 1, 1] = 1 ∧
      strictRemovalUpperBound [1, 1, 1] ≠ assignmentSearchBound [1, 1, 1] := by
  decide

theorem totalLiveCount_eq_sum_sizes (holes : HoleDomains) :
    totalLiveCount holes = listSum (holeSizes holes) :=
  rfl

private theorem listSum_map_length_cons (h : List Candidate) (hs : HoleDomains) :
    listSum (holeSizes (h :: hs)) =
      h.length + listSum (holeSizes hs) := by
  simp [holeSizes, listSum]

theorem closePassHoles_total_le
    (holes : HoleDomains) (results : List QueryResult) :
    totalLiveCount (closePassHoles holes results) ≤ totalLiveCount holes := by
  induction holes with
  | nil => simp [closePassHoles, totalLiveCount, holeSizes, listSum]
  | cons h hs ih =>
      have hpass : (closePass h results).length ≤ h.length :=
        closePass_length_le h results
      have ih' := ih
      simp only [closePassHoles, totalLiveCount, listSum_map_length_cons] at ih' ⊢
      exact Nat.add_le_add hpass ih'

def totalRemovedAcrossHoles (holes : HoleDomains) : List (List QueryResult) → Nat
  | [] => 0
  | batch :: rest =>
      (totalLiveCount holes - totalLiveCount (closePassHoles holes batch)) +
        totalRemovedAcrossHoles (closePassHoles holes batch) rest

theorem totalRemovedAcrossHoles_le_sum
    (holes : HoleDomains) (batches : List (List QueryResult)) :
    totalRemovedAcrossHoles holes batches ≤ totalLiveCount holes := by
  induction batches generalizing holes with
  | nil => simp [totalRemovedAcrossHoles]
  | cons batch rest ih =>
      simp only [totalRemovedAcrossHoles]
      have hle : totalLiveCount (closePassHoles holes batch) ≤ totalLiveCount holes :=
        closePassHoles_total_le holes batch
      have ih' := ih (closePassHoles holes batch)
      calc
        totalLiveCount holes - totalLiveCount (closePassHoles holes batch) +
            totalRemovedAcrossHoles (closePassHoles holes batch) rest
          ≤ totalLiveCount holes - totalLiveCount (closePassHoles holes batch) +
              totalLiveCount (closePassHoles holes batch) := Nat.add_le_add_left ih' _
        _ = totalLiveCount holes := Nat.sub_add_cancel hle

theorem totalRemovedAcrossHoles_le_strictRemovalBound
    (holes : HoleDomains) (batches : List (List QueryResult)) :
    totalRemovedAcrossHoles holes batches ≤
      strictRemovalUpperBound (holeSizes holes) := by
  simpa [strictRemovalUpperBound, totalLiveCount] using
    totalRemovedAcrossHoles_le_sum holes batches

private theorem listSum_ge_length_of_pos :
    ∀ (sizes : List Nat), (∀ d ∈ sizes, 1 ≤ d) → sizes.length ≤ listSum sizes
  | [], _ => by simp [listSum]
  | d :: ds, h => by
      have hd : 1 ≤ d := h d (by simp)
      have hds : ∀ x ∈ ds, 1 ≤ x := fun x hx => h x (by simp [hx])
      have ih := listSum_ge_length_of_pos ds hds
      simp [listSum]
      omega

theorem allNonempty_live_ge_hole_count
    (holes : HoleDomains) (h : allNonempty holes) :
    holes.length ≤ totalLiveCount holes := by
  have hpos : ∀ d ∈ holeSizes holes, 1 ≤ d := by
    intro d hd
    obtain ⟨live, hlive, rfl⟩ := List.mem_map.mp hd
    have hne : live ≠ [] := h live hlive
    cases live with
    | nil => exact (hne rfl).elim
    | cons _ _ => exact Nat.succ_le_succ (Nat.zero_le _)
  simpa [totalLiveCount, holeSizes] using listSum_ge_length_of_pos (holeSizes holes) hpos

private theorem listSum_map_pred_eq_sum_sub_length :
    ∀ (sizes : List Nat),
      (∀ d ∈ sizes, 1 ≤ d) →
        listSum (sizes.map fun d => d - 1) =
          listSum sizes - sizes.length
  | [], _ => rfl
  | d :: ds, h => by
      have hd : 1 ≤ d := h d (by simp)
      have hds : ∀ x ∈ ds, 1 ≤ x := fun x hx => h x (by simp [hx])
      have ih := listSum_map_pred_eq_sum_sub_length ds hds
      have hge := listSum_ge_length_of_pos ds hds
      simp [listSum, List.map]
      have : d - 1 + (listSum ds - ds.length) =
          d + listSum ds - (ds.length + 1) := by omega
      calc
        d - 1 + listSum (ds.map fun x => x - 1)
          = d - 1 + (listSum ds - ds.length) := by rw [ih]
        _ = d + listSum ds - (ds.length + 1) := this

theorem excess_eq_total_sub_holes
    (holes : HoleDomains) (h : allNonempty holes) :
    nonemptyInvariantUpperBound (holeSizes holes) =
      totalLiveCount holes - holes.length := by
  have hpos : ∀ d ∈ holeSizes holes, 1 ≤ d := by
    intro d hd
    obtain ⟨live, hlive, rfl⟩ := List.mem_map.mp hd
    have hne : live ≠ [] := h live hlive
    cases live with
    | nil => exact (hne rfl).elim
    | cons _ _ => exact Nat.succ_le_succ (Nat.zero_le _)
  simpa [nonemptyInvariantUpperBound, totalLiveCount, holeSizes] using
    listSum_map_pred_eq_sum_sub_length (holeSizes holes) hpos

/--
  When every intermediate hole domain stays nonempty, cumulative removals are
  bounded by the nonempty-invariant lattice height ``Σ (d_i - 1)``.
  Preconditions: initial and final states are all-nonempty; live count only
  shrinks; hole cardinality is preserved (same holes).
-/
theorem removals_le_nonempty_invariant
    (initial final : HoleDomains)
    (hinit : allNonempty initial)
    (hfinal : allNonempty final)
    (hlen : initial.length = final.length)
    (_hmono : totalLiveCount final ≤ totalLiveCount initial) :
    totalLiveCount initial - totalLiveCount final ≤
      nonemptyInvariantUpperBound (holeSizes initial) := by
  have hge : final.length ≤ totalLiveCount final :=
    allNonempty_live_ge_hole_count final hfinal
  have hexcess := excess_eq_total_sub_holes initial hinit
  have : totalLiveCount initial - totalLiveCount final ≤
      totalLiveCount initial - final.length :=
    Nat.sub_le_sub_left hge _
  calc
    totalLiveCount initial - totalLiveCount final
      ≤ totalLiveCount initial - final.length := this
    _ = totalLiveCount initial - initial.length := by rw [hlen]
    _ = nonemptyInvariantUpperBound (holeSizes initial) := by rw [← hexcess]

/-- Concrete fixture: domains [2, 3] → Σ = 5, Σ(d-1) = 3, ∏ = 6. -/
theorem fixture_two_three_stabilization_bounds :
    strictRemovalUpperBound [2, 3] = 5 ∧
      nonemptyInvariantUpperBound [2, 3] = 3 ∧
      assignmentSearchBound [2, 3] = 6 := by
  decide

theorem fixture_flat_total_removed_le_live :
    let live : List Candidate := [0, 1, 2, 3]
    let r0 : QueryResult :=
      { hole := 0, candidate := 1, verdict := .unsupported, replayOk := true }
    let r1 : QueryResult :=
      { hole := 0, candidate := 3, verdict := .unsupported, replayOk := true }
    totalRemovedAcross live [[r0], [r1]] ≤ live.length ∧
      totalRemovedAcross live [[r0], [r1]] = 2 := by
  decide

/-- Fixed points are not claimed maximal; coverage is recorded beside the fixed point. -/
theorem fixed_point_not_automatic_maximum
    (live : List Candidate) (results : List QueryResult)
    (h : isFixedPoint live results) :
    isFixedPoint live results ∧ batchCoversRemovable live results :=
  ⟨h, batchCoversRemovable_holds live results⟩

end LeverProofLean.ExactClosure
