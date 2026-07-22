# E864-E865: opaque-marker validity ownership

E864 replayed the retained E852 checkpoint on the unchanged E842 smoke subset
under evaluation harness v50. All three generated programs used only canonical
contiguous `:slot_N` identities, and exact fidelity plus contract precision and
recall were 1.0000. The diagnostic `placeholder_validity` nevertheless reported
0.6000 because its independent “well formed” test still required a dot, rewarding
the prohibited semantic namespace format.

The evaluator now reuses the canonical marker predicate owned by the data
contract. The legacy `placeholder_fidelity_normalized` result field remains for
schema compatibility but no longer strips or credits semantic namespaces. This
is a harness/test-contract correction; no model conversion, inference, training,
or checkpoint change is involved.

E865 repeated the identical local CPU recipe under harness v51. All three
prediction hashes exactly match E864. `placeholder_validity` rose from 0.6000 to
1.0000; parse, meaning-v1, strict meaning-v2, structure (0.6589), component
recall (0.7500), fidelity (1.0000), reward (0.9490), and zero timeout/fallback
were unchanged. AgentV remained 0/1 because this is smoke `n=3`, so the result is
diagnostic only and makes no ship or promotion claim.

Canonical evidence:
[`iter-e864-e865-opaque-marker-validity-20260722.json`](iter-e864-e865-opaque-marker-validity-20260722.json).
