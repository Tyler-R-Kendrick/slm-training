# E763-E764 — symbol-only held-out data and request-bound fallback

**Date:** 2026-07-22
**Decision:** retain test-data v5 and TwoTower v222; do not promote the checkpoint
**Evidence:** [`iter-e763-e764-symbol-only-heldout-fallback-20260722.json`](iter-e763-e764-symbol-only-heldout-fallback-20260722.json)

The first strict held-out replay did not reach model evaluation: the immutable
legacy `remediated` eval snapshot contains free-form `TabItem.value` strings.
The canonical seed audit found the same class in form, tabs, and settings
records. Test-data v5 replaces those values with declared markers, adds the
seed file to component-version coverage, and makes two fixtures structurally
disjoint by adding `Separator` nodes. The leakage gate was not weakened.

The retained published snapshot,
`e763_symbol_only_eval_r2_20260722`, has 19 records across all five suites,
zero output-contract errors, zero undeclared markers, and 37 train-overlap
rejections. Sanitization made no literal-to-marker substitutions and used no
fallbacks; the canonical sources are already symbol-only. The failed
preflight build that detected two train-topology collisions emitted no
snapshot and is not evidence.

E763 then exposed the actual harness defect. Four failed decodes were finalized
as the hard-coded `root = Button(":cta.label")`, even though `:cta.label` was
not in any request contract. TwoTower v222 removes all canned marker strings.
Its certified fallback uses the first marker declared by the request, or
`Separator()` when no marker contract exists. It does not synthesize the whole
expected tree, so weak model behavior remains visible.

The matched E764 local replay confirms the invariant: contract precision rises
from 0.2 to 1.0, and all five predictions use only request-declared markers.
Parse remains 1.0, but strict meaningful-v2 remains 0, structural similarity is
0.2155, component recall is 0.2286, and AgentV is 0/1. This is an honest
negative quality result, not a ship claim. Four certified fallbacks remain;
the next work is improving the constrained model path, not expanding fallback
templates.

Both successful commands ran locally in under 26 seconds beneath the
110-second command guard. No remote workflow ran. No checkpoint was created or
synced.
