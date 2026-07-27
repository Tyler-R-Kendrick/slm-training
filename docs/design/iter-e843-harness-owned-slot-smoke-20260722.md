# E843: harness-owned slot smoke

## Outcome

The unchanged E832 checkpoint was evaluated locally against the new canonical
E842 eval default with the same semantic-plan weights as E837. Smoke `n=3`
matched E837 on every quality metric: parse and meaningful-program rates 1.0,
strict binding-aware meaning 1.0, placeholder fidelity 1.0, structural
similarity 0.6033, component recall 0.6667, and reward 0.9490. Decode recorded
zero timeouts, zero fallback, and p50/p95 latency of 3.03/3.71 seconds.

This confirms that moving marker conversion and inventory ownership out of the
models did not change the established smoke result. The run used the
`e842_harness_owned_slots_v1` corpus through `DEFAULT_EVAL_DATA_DIR`; the
request harness supplied the structured opaque inventory. AgentV executed one
gate case and failed it (mean score 0.5), so this remains diagnostic evidence,
not a ship or checkpoint-promotion claim.

No training, checkpoint creation, remote compute, sync, or deployment occurred.
