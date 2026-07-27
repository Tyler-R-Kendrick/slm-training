# SLM-293 — local property-role saturation control

Status: capacity control complete; legacy connector dispositioned non-causal.

This bounded CPU run froze the base generator and trained only existing auxiliary
heads on 2 declared train records for 24 steps. It is
not a held-out evaluation, a checkpoint, or a ship claim.

| Factor | Head accuracy | Required live decode counter | Head disposition | Connector disposition |
| --- | ---: | --- | --- | --- |
| inventory | 0.83 | component_inventory_choice_changes | head_not_saturated | non_causal_not_on_decoder_path |
| topology | 0.67 | component_edge_choice_changes | head_not_saturated | non_causal_not_on_decoder_path |
| cardinality | 1.00 | binder_arity_choice_changes | capacity_saturated_decoder_control_pending | non_causal_not_on_decoder_path |
| property_ownership | 1.00 | slot_component_choice_changes | capacity_saturated_decoder_control_pending | non_causal_not_on_decoder_path |
| binder | 1.00 | binder_component_plan_choice_changes | capacity_saturated_decoder_control_pending | non_causal_not_on_decoder_path |
| reference_edge | 1.00 | root_reference_arity_choice_changes | capacity_saturated_decoder_control_pending | non_causal_not_on_decoder_path |

The paired control executes learned, gold-substituted, shuffled across examples,
and zeroed values at parser-derived eligible factor choices. The next held-out
decode must collect all four controls' decoder eligible-position argmax-change
counters, then report meaningful-v2 and strict-program outcomes. Until then
every factor remains non-promotable.

The explicitly configured legacy `semantic_connector=low_rank` installs no
`SemanticConnector` module in TwoTower. Its eligible decode positions and
choice changes are therefore exactly zero, and its structural meaning-v2 delta
is zero: **non-causal, not on the decoder path**. This is a connector
disposition, not a claim that the separate TwoTower auxiliary heads are ready
to promote.
