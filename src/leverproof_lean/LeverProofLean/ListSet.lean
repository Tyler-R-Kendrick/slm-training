/-
  Finite-set algebra over lists.

  Self-contained membership model: two lists represent the same set when they
  have the same members. No Mathlib; only Lean core List / Nat primitives.
-/

namespace LeverProofLean

/-- Set inclusion up to membership (list order and duplicates ignored). -/
def Subset (xs ys : List Nat) : Prop :=
  ∀ x, x ∈ xs → x ∈ ys

/-- Set equality up to membership. -/
def SetEq (xs ys : List Nat) : Prop :=
  Subset xs ys ∧ Subset ys xs

theorem Subset.refl (xs : List Nat) : Subset xs xs := by
  intro x hx
  exact hx

theorem Subset.trans {xs ys zs : List Nat}
    (hxy : Subset xs ys) (hyz : Subset ys zs) : Subset xs zs := by
  intro x hx
  exact hyz x (hxy x hx)

theorem SetEq.refl (xs : List Nat) : SetEq xs xs :=
  ⟨Subset.refl xs, Subset.refl xs⟩

theorem SetEq.symm {xs ys : List Nat} (h : SetEq xs ys) : SetEq ys xs :=
  ⟨h.2, h.1⟩

theorem SetEq.trans {xs ys zs : List Nat}
    (hxy : SetEq xs ys) (hyz : SetEq ys zs) : SetEq xs zs :=
  ⟨Subset.trans hxy.1 hyz.1, Subset.trans hyz.2 hxy.2⟩

theorem Subset.append_left (xs ys : List Nat) : Subset xs (xs ++ ys) := by
  intro x hx
  exact List.mem_append.mpr (Or.inl hx)

theorem Subset.append_right (xs ys : List Nat) : Subset ys (xs ++ ys) := by
  intro x hx
  exact List.mem_append.mpr (Or.inr hx)

theorem mem_append_iff (x : Nat) (xs ys : List Nat) :
    x ∈ xs ++ ys ↔ x ∈ xs ∨ x ∈ ys :=
  List.mem_append

/-- Members of `xs` that also belong to `ys`. -/
def inter (xs ys : List Nat) : List Nat :=
  xs.filter (fun x => decide (x ∈ ys))

theorem mem_inter_iff (x : Nat) (xs ys : List Nat) :
    x ∈ inter xs ys ↔ x ∈ xs ∧ x ∈ ys := by
  simp [inter, List.mem_filter]

theorem inter_subset_left (xs ys : List Nat) : Subset (inter xs ys) xs := by
  intro x hx
  exact (mem_inter_iff x xs ys).mp hx |>.1

theorem inter_subset_right (xs ys : List Nat) : Subset (inter xs ys) ys := by
  intro x hx
  exact (mem_inter_iff x xs ys).mp hx |>.2

theorem inter_mono_left {xs ys zs : List Nat} (h : Subset xs ys) :
    Subset (inter xs zs) (inter ys zs) := by
  intro x hx
  have hx' := (mem_inter_iff x xs zs).mp hx
  exact (mem_inter_iff x ys zs).mpr ⟨h x hx'.1, hx'.2⟩

/-- Members of `xs` that do not belong to `ys`. -/
def diff (xs ys : List Nat) : List Nat :=
  xs.filter (fun x => decide (x ∉ ys))

theorem mem_diff_iff (x : Nat) (xs ys : List Nat) :
    x ∈ diff xs ys ↔ x ∈ xs ∧ x ∉ ys := by
  simp [diff, List.mem_filter]

theorem diff_subset (xs ys : List Nat) : Subset (diff xs ys) xs := by
  intro x hx
  exact (mem_diff_iff x xs ys).mp hx |>.1

theorem diff_disjoint (xs ys : List Nat) (x : Nat)
    (hx : x ∈ diff xs ys) : x ∉ ys :=
  (mem_diff_iff x xs ys).mp hx |>.2

/-- List union: concatenation (membership is the set union). -/
def union (xs ys : List Nat) : List Nat :=
  xs ++ ys

theorem mem_union_iff (x : Nat) (xs ys : List Nat) :
    x ∈ union xs ys ↔ x ∈ xs ∨ x ∈ ys := by
  simp [union, List.mem_append]

theorem union_subset_of_subsets {xs ys zs : List Nat}
    (hxs : Subset xs zs) (hys : Subset ys zs) : Subset (union xs ys) zs := by
  intro x hx
  cases (mem_union_iff x xs ys).mp hx with
  | inl h => exact hxs x h
  | inr h => exact hys x h

theorem subset_union_left (xs ys : List Nat) : Subset xs (union xs ys) :=
  Subset.append_left xs ys

theorem subset_union_right (xs ys : List Nat) : Subset ys (union xs ys) :=
  Subset.append_right xs ys

theorem union_mono_right {xs a b : List Nat} (h : Subset a b) :
    Subset (union xs a) (union xs b) := by
  intro x hx
  cases (mem_union_iff x xs a).mp hx with
  | inl hx => exact (mem_union_iff x xs b).mpr (Or.inl hx)
  | inr hx => exact (mem_union_iff x xs b).mpr (Or.inr (h x hx))

theorem union_idempotent_mem (xs : List Nat) :
    SetEq (union xs xs) xs := by
  constructor
  · intro x hx
    cases (mem_union_iff x xs xs).mp hx with
    | inl h => exact h
    | inr h => exact h
  · exact subset_union_left xs xs

end LeverProofLean
