/-
  Training-sample adequacy bounds (data-volume signal for the climb loop).

  Two exact Nat-arithmetic quantities consumed through the bound_ast/v1
  registry (`bound.sample_size.coverage_lower.v1`,
  `bound.sample_size.capacity_upper.v1`):

  * `coverageLowerBound k n wMin` — records required for every tracked
    component to accumulate `k` witnesses, projecting the observed witness
    rate of the scarcest component (`wMin` witnesses across `n` records).
    The projection assumption — synthesis keeps witnessing the scarcest
    component at its observed rate — is declared by the caller; the
    arithmetic here is proved, the rate persistence is not.

  * `capacityUpperBound params bitsPerParam bitsPerExample` — examples whose
    total description length fits the parameter budget under a declared
    bits-per-parameter capacity prior. The prior interval is an assumption
    supplied by the caller; only the budget arithmetic is proved.

  Neither bound is a generalization or quality claim, and neither is a
  wall-clock claim.
-/

namespace LeverProofLean.SampleAdequacy

/-- Records required for `k` witnesses of a component observed `wMin` times
    across `n` records, at the observed rate: `⌈k·n / wMin⌉` via Nat
    ceiling division. -/
def coverageLowerBound (k n wMin : Nat) : Nat :=
  (k * n + wMin - 1) / wMin

/-- Examples affordable by `params` parameters at `bitsPerParam` capacity
    when each example costs `bitsPerExample` description bits (floor). -/
def capacityUpperBound (params bitsPerParam bitsPerExample : Nat) : Nat :=
  params * bitsPerParam / bitsPerExample

/-- The coverage bound suffices: training `coverageLowerBound k n wMin`
    records at the observed rate accumulates at least the `k·n` witness-mass
    the target demands (scarcest component, rate `wMin/n` per record). -/
theorem coverageLowerBound_covers (k n wMin : Nat) (h : 0 < wMin) :
    k * n ≤ wMin * coverageLowerBound k n wMin := by
  unfold coverageLowerBound
  have h1 := Nat.div_add_mod (k * n + wMin - 1) wMin
  have h2 := Nat.mod_lt (k * n + wMin - 1) h
  omega

/-- The capacity bound never claims more description bits than the
    parameter budget stores under the declared prior. -/
theorem capacityUpperBound_within_budget
    (params bitsPerParam bitsPerExample : Nat) :
    capacityUpperBound params bitsPerParam bitsPerExample * bitsPerExample
      ≤ params * bitsPerParam := by
  unfold capacityUpperBound
  exact Nat.div_mul_le_self (params * bitsPerParam) bitsPerExample

-- Registry parity anchors (mirrored by bound_ast_parity_fixtures.v1.json).
#guard coverageLowerBound 4 101 2 = 202
#guard coverageLowerBound 4 101 101 = 4
#guard coverageLowerBound 4 101 1 = 404
#guard capacityUpperBound 1000 3 60 = 50
#guard capacityUpperBound 1000 6 60 = 100

end LeverProofLean.SampleAdequacy
