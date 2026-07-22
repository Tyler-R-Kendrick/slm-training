# E731 lexer-native root-reference arity

**Date:** 2026-07-22  
**Decision:** retain the capability; reject the checkpoint and close weights 0/1/2  
**Evidence:** [`iter-e731-lexer-root-arity-symbol-only-20260722.json`](iter-e731-lexer-root-arity-symbol-only-20260722.json)

E731 implements the capability E726 proved was missing: root-reference arity
now has grammar-native lexer supervision and compiler-tree/restricted
continue-versus-stop scoring. The canonical lever registry advertises this
path, while unsupported combinations still fail before artifacts. Tests prove
target extraction, gradient flow, both compiler integrations, and capability
discovery.

The matched local CPU train adds root-arity loss/decode weight 1 to E723's
symbol-only plan+edge+slot-owner recipe. It completes 140 steps in 82.20 seconds
under `max_wall_minutes=2`. Final root-arity loss is 1.0957 with 0.5 accuracy
over two active rows; primary reconstruction and slot-owner metrics exactly
match E723. The local-only checkpoint SHA is
`bff1e0e6b07f3063c59b6549c121b4ceb38e7f0a5a90f093783673bcac2fbb88`.

On the same clean three-record strict smoke diagnostic, decode weights 0, 1,
and 2 are prediction- and metric-identical: parse 1.0, meaning-v1 0.6667,
strict-v2 0.0, fidelity 0.5278, structure 0.5614, recall 0.4167, and reward
0.8073. Weights 1 and 2 each apply six times but change zero choices. No arm
times out; AgentV is 0/1.

Keep the generalized mechanism available and default-off. Reject, do not
upload, and do not promote this scratch checkpoint. The next arm should not
increase this scalar; it needs a signal that changes which additional bound
components become reachable, not another continue/stop multiplier.
