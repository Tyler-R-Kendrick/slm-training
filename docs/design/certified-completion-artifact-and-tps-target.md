# Certified completion artifact and causal TPS target

## Decision

OpenUI ships a proof-carrying, request-independent completion artifact beside
the code, and the decoder reuses it only after a small checker proves equality
with the live grammar and tokenizer authority. Scope and semantic constraints
remain a request-local overlay.

The DSL Packs dashboard also exposes a normative throughput interval calculated
by one backend model. It uses the declared whole-model parameter count,
grammar-derived forward/oracle work, and an explicit hardware profile. Observed
runs are never inputs; they are joined afterward only when their throughput
profile is declared.

This implements a certified-checker design rather than a Rocq/Lean
formalization. The proof boundary is executable, deterministic, and narrow
enough to audit.

## Why the old target was wrong

The previous page passed the grammar description's 299–712 parameter
information-capacity estimate into a dense-model Roofline equation. That
estimate answers “how many parameters might store this grammar?” It does not
describe the parameters read by the serving model. Dividing compute and memory
rates by hundreds of parameters therefore produced a floor with no causal
relationship to current inference.

It also multiplied by logical cores in the same value used as a latency target,
calculated the formula in React, and compared profile-untagged observations.
The replacement keeps four quantities distinct:

1. grammar information and ambiguity;
2. whole-model neural work;
3. single-stream latency throughput;
4. aggregate multi-stream throughput.

## Artifact boundary

The committed pair is:

- `resources/decode/openui_completion_v1.safetensors`;
- `resources/decode/openui_completion_v1.manifest.json`.

The safe binary contains integer-only arrays for the canonicalized LALR control
graph and tokenizer-to-terminal projection. The JSON manifest binds the
artifact, grammar bytes, grammar terminal definitions, tokenizer layout,
checker schema, dimensions, and excluded dynamic authority with SHA-256.

Lark's raw LALR state numbers are deliberately not serialized as identity:
they vary with process hash iteration. The builder applies deterministic
partition refinement to the labeled transition graph. Shift targets are
represented by their stable behavioral color and reductions by a canonical rule
id. The checker independently performs the same canonicalization over the live
parser.

New lexer-native TwoTower checkpoints embed the artifact schema, checker
schema, artifact digest, grammar digest, and tokenizer-authority digest in both
the checkpoint payload and `.meta.json`. Saving fails closed if the committed
file and manifest disagree, and loading a bound checkpoint rejects any mismatch
with the installed artifact. Existing checkpoints remain loadable; they simply
have no artifact identity and therefore cannot claim that binding.

The following facts are never baked in:

- runtime symbols and caller spellings;
- binder/state/placeholder visibility;
- literal bodies;
- corpus/request macro expansions;
- semantic state and schema position.

Those facts continue through `CompletionSession` and `SemanticState`.

## Checker obligations

For artifact `A`, live grammar `G`, tokenizer `T`, and request-local semantic
overlay `S`, acceptance requires:

1. `hash(A)` equals the manifest digest;
2. `hash(G)` and the canonical terminal digest equal the manifest;
3. `hash(T)` over ids, kinds, slot counts, version, binding mode, and special
   ids equals the manifest;
4. decoding the safe tensors reproduces their declared shapes;
5. the artifact's tokenizer projection equals
   `dsl_direct_terminal_map(T, G)` exactly;
6. the canonical control arrays and metadata equal a fresh compilation of
   `G`.

Only (5) is consumed to accelerate direct feeds in v1; Lark remains the live
parser. This makes the safety theorem small:

> If the checker accepts, every artifact-projected direct feed is exactly a
> direct feed already authorized by the live grammar/tokenizer projection.

The combined production domain remains `static(G,T) ∩ dynamic(S)`. Dynamic
authority can only tighten the static domain. Exact singleton bypass therefore
retains I1/I2, and final grammar certification retains I6. Missing, stale,
corrupt, unsupported, or unequal artifacts use the exact live construction;
they never widen the legal set.

`tests/test_dsl/test_completion_artifact.py` checks live equivalence,
cross-process-stable committed data, deterministic rebuilding, runtime
selection, and corruption rejection. Existing completion-kernel differential
and singleton-forward tests continue to own semantic coverage and zero-forward
proofs.

## Causal performance target

For parameters `P`, neural forwards `F`, output tokens `L`, oracle steps `O`,
streams `N`, and declared hardware rates:

```text
t_compute = P * ops_per_parameter * F / (compute_utilization * peak_compute)
t_memory  = P * bytes_per_parameter * F / (bandwidth_utilization * bandwidth)
t_serial  = O * (dispatch_step + oracle_ops_per_step * oracle_operation_cost)

TPS = N * L / (max(t_compute, t_memory) + t_serial)
```

The floor uses the conservative grammar-derived structural-forward bound; the
ceiling uses its work/span lower bound. `N=1` is always reported separately
from the aggregate `N>1` profile. Timeouts are policy caps, never throughput
inputs. `oracle_ops_per_step` is one table transition, the number of 64-bit
words needed for the declared component/structural-symbol masks, and the
grammar's maximum production alternatives. Total artifact rows do not enter
the per-step term because checked table access is O(1).

Inputs and causal levers are exposed in the API:

| Input | Source | Lever |
| --- | --- | --- |
| `P` | latest declared checkpoint/model metadata | smallest sufficient model; growth pays `EG_params` |
| `F` | AST work/span bounds | I1/I2 singleton bypass, I3 verified spans, I4 row batching |
| bytes/parameter | declared target profile | quantization/cache that preserves exact authority |
| compute/bandwidth rates and utilization | detected plus explicitly labeled planning assumptions | target device/runtime/kernel |
| serial dispatch and oracle operation | DSL alternatives/symbol-mask width × explicit cost assumptions | checked lookup, bitsets, and cheaper dynamic overlay |
| streams | explicit profile | independent request batching; aggregate only |

The dashboard reports old-code mismatches beside these levers. Existing
performance rows have no declared single-stream/aggregate profile, so their
gap is shown as the range spanning both target floors. That makes the scale of
the mismatch visible without silently selecting the wrong profile; future
writers should emit one exact profile.

## Prior work and adversarial assessment

| Work | Useful result | Boundary for this implementation |
| --- | --- | --- |
| [XGrammar](https://arxiv.org/abs/2411.15100) and Outlines | grammar/tokenizer indexing can amortize constrained decoding | syntax indexing does not prove request-local scope |
| [SynCode](https://arxiv.org/abs/2403.01632) and [DOMINO](https://arxiv.org/abs/2403.06988) | offline lookup, token alignment, and minimally invasive masking matter | their autoregressive setting does not transfer a full forest theorem to diffusion |
| llguidance | precomputation may trade runtime work for startup and memory | motivates the narrow, measured artifact instead of serializing arbitrary prefixes |
| PICARD and grammar refinement | live incremental checks catch semantic/schema invalidity | justifies keeping semantic authority outside the file |
| Mündler et al. and constrained diffusion | CFG intersection/witness reasoning applies to diffusion | bounded or partial witnesses must remain `UNKNOWN`, not “complete” |
| [Jourdan, Pottier, and Leroy](https://cambium.inria.fr/~fpottier/publis/jourdan-leroy-pottier-validating-parsers.pdf) | validating an untrusted automaton with a small checker can provide a trustworthy result | motivates treating the builder as untrusted and the runtime checker as authority; our Python checker is not their Coq proof |
| Roofline | compute and bandwidth form separate ceilings | cannot identify utilization or serial overhead from DSL size |
| Orca and Sarathi-Serve | batching/scheduling changes aggregate throughput and latency differently | requires separate profiles rather than a core multiplier in one TPS value |

The main adversarial failures are stale artifacts, malicious tensor offsets,
tokenizer-layout drift, grammar drift, unstable parser ids, dynamic-authority
leakage, profile mixing, and circular calibration from observations. The current
format/checker rejects the first five, structurally excludes dynamic facts,
separates profiles, and keeps observed metrics outside the target function.

## Claim-family split (executor vs circuit vs topology heads)

Architecture disposition for constrained diffusion and topology lives in
[adr-constrained-diffusion-topology-split.md](adr-constrained-diffusion-topology-split.md).
This artifact doc remains the certificate/TPS home; the ADR owns:

- **Family A (engine executor):** certified static control + Γ leaf filters,
  E1 multiset parity, E4 forced macros, E8 zero-forward, E9 memo-reuse labeling;
- **Family B (diffusion circuit):** residual multi-hole *predicate* later —
  not a first milestone for this artifact;
- **Family C (model-facing topology heads):** frozen pending anti-E237.

Production domains for E1 are exposed via
`slm_training.dsl.grammar.fastpath.static_control_domain`. Warm hard-prefix
~89.8× is **`request_local_memo_reuse`** only — never AOT cold.

## Remaining work

- Add an explicit target-profile registry for named CPU/GPU serving classes;
  current unidentified rates remain visibly labeled assumptions.
- Tag future performance evidence `single_stream` or `aggregate` at its writer
  so gap analysis becomes comparable.
- Measure artifact load and checker overhead in a preregistered performance run
  before claiming a speedup (E3). This change proves reuse and authority parity,
  not a whole-model latency win.
- Consume the certified control table directly only after a separately checked
  parser adapter proves identical accept/reduce behavior **and** E1 stays green
  (executable static control successor — not a rebrand of today's loader).
- Escalate to a proof assistant only if the checker or artifact language grows
  beyond this auditable finite relation.
