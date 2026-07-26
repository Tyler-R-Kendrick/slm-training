# AP-034 conditional latent-prior admission disposition

SLM-335 asks whether a prompt-conditioned multi-step prior can approach the
oracle latent before the one-step AP-035 experiment starts. This is an
admission decision, not permission to condition a prior on the target plan or
to rename a reconstruction proxy as semantic quality.

## Audit (2026-07-26)

The completed predecessors provide a program-factor bottleneck but not the two
interfaces that AP-034 needs:

- `cap2_rate_distortion_sweep.py` and
  `cap2_latent_causal_audit.py` consume tensorized `SemanticPlanV1` factors
  and report factor-reconstruction MSE / a strict factor-MSE proxy. Their
  fixture loader returns plans, not immutable prompt-to-plan pairs.
- `ContinuousLatentCodec` encodes factor tensors; it has no prompt/context
  input, conditional density, flow/diffusion sampler, or latent-prior API.
- `SemanticPlanV1.identity` records only a `prompt_context_hash`, never the
  prompt text or an independently usable prompt representation.
- The AP-030 and AP-031 evidence explicitly records that no verified
  latent-to-program decoder exists. Consequently
  `binding_aware_meaningful_v2` and binder/reference F1 cannot be evaluated
  for a sampled latent without inventing an unaudited decode path.

Using the factor tensor (or a field derived from it) as the AP-034 condition
would leak the oracle target into `p(z | condition)`. Calling the existing
factor MSE proxy `meaning-v2` or binder/reference F1 would likewise make the
acceptance comparison false. Neither shortcut is admitted.

## Disposition

**Conditional-prior integration is unavailable; no model, connector, or
training path was added.** The completed AP-030/AP-031 controls stay
default-off and diagnostic. No local training, checkpoint, metric curve,
human-rating gate, remote/HF operation, promotion, or ship claim is made by
this disposition.

The required successor admission artifacts are:

1. An immutable, train-only prompt-to-`SemanticPlanV1` manifest with split,
   provenance, and held-out leakage audits, whose condition cannot be derived
   from target factors or oracle plan fields.
2. A constrained, replay-verified latent-to-program decode bridge that emits
   strings accepted by `binding_aware_meaningful_v2` and the binder/reference
   evaluator, without widening legal decode authority.
3. A matched oracle / nearest-encoder / random / conditional-prior protocol
   over that same held-out manifest, with all 2/4/8/16/32-step arms, AgentV
   results, and the existing multi-suite gates left unchanged.

Only after those artifacts exist may a future implementation add the requested
default-off conditional prior and assess the 0.10 oracle margin. Until then,
the AP-035 dependency remains unsupported rather than inferred from fixture
factor reconstruction.
