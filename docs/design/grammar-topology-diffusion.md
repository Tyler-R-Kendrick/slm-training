# Grammar-topology diffusion

**Status:** implemented experimental replacement for the former fixed-canvas
`grammar_diffusion` plug-in. It is not a ship claim. X9-X15 must establish quality,
topology, trace, and efficiency evidence before promotion.

## Why the state space changed

The former model denoised a preallocated production sequence. A mask represented
one unknown value at one existing position; `block_size` grouped positions but did
not create topology. The replacement operates on a bounded, changing production
tree. A mask is a typed work item (`document`, `statement`, `expression`,
`component`, `list`, or `leaf`) whose reverse transition may expand, keep, delete,
contract, or stop.

This is an OpenUI-specific adaptation of insertion/deletion and hierarchical
generation ideas, not a faithful MaskGIT implementation. Relevant lineage includes
[MaskGIT](https://arxiv.org/abs/2202.04200),
[MDLM](https://arxiv.org/abs/2406.07524),
[Block Diffusion](https://arxiv.org/abs/2503.09573),
[Insertion Transformer](https://arxiv.org/abs/1902.03249),
[Levenshtein Transformer](https://arxiv.org/abs/1905.11006), and
[Diffusion Forcing](https://arxiv.org/abs/2407.01392). The newer
[Deletion-Insertion Diffusion](https://arxiv.org/abs/2603.23507) and
[Multi-Block Diffusion](https://arxiv.org/abs/2606.29215) are adjacent evidence for
trans-dimensional state and bounded concurrent work. See
[research-lineage.md](research-lineage.md) for fidelity labels.

## State, corruption, and reverse process

Each node has a persistent ID plus production, slot, parent, depth, and sibling
coordinates. The denoiser uses production, node-type, parent-type, depth, and
sibling embeddings; it has no learned absolute sequence-position embedding.

Training parses each gold production stream into a tree, then applies online
corruptions:

- collapse a subtree into one typed mask;
- inject a deletable child;
- corrupt a visible node and supervise contraction;
- optionally give different nodes/depths different noise rates.

The reverse model exposes production, slot, action, arity, critic, and confidence
heads. Decode begins with one active document mask. Proposed rewrites in a phase are
collected before mutation and applied synchronously. Production legality is filtered
by node type. A bounded active-node buffer performs local passes, with periodic
whole-tree synchronization for global consistency and contraction. Hard node,
active-node, arity, depth, and phase limits fail closed instead of silently falling
back to a canned program.

The practical state space is therefore bounded and ragged rather than an unbounded
graph. This keeps accelerator batching possible while allowing local topology to
grow and shrink.

## Training and evaluation signals

Training logs the component losses and head diagnostics alongside ordinary loss:

| Signal | Meaning |
| --- | --- |
| `action_loss`, `action_macro_f1` | expand/keep/delete/contract/stop supervision |
| `production_loss`, `production_head_accuracy` | production chosen for collapsed nodes |
| `arity_loss`, `arity_head_accuracy` | child count chosen for expansion |
| `slot_loss` | honest slot-contract pointer prediction |
| `critic_loss`, `critic_ece` | accept/contract calibration against corrupted-tree labels |
| `active_nodes`, `materialized_nodes` | train-time topology workload |

Generated output adds production-sequence accuracy, topology arity accuracy,
AST node/edge F1, tree-edit similarity, phase count, peak active nodes, node passes,
expansion/deletion/contraction counts, budget failures, and steps to first valid
program. Teacher-forced head metrics are computed after generation and never supply
gold fields to decode.

The topology matrix ranks candidates with a quality-heavy composite:

```text
Q = weighted honest parse/fidelity/structure/reward
T = mean(AST node F1, AST edge F1, tree-edit similarity)
R = mean(action macro-F1, generated production accuracy,
         generated arity accuracy, 1 - critic ECE)
E = bounded node-pass efficiency

topology_composite = 0.45 Q + 0.25 T + 0.20 R + 0.10 E
```

All underlying metrics remain visible. The composite cannot replace ship gates;
it only orders topology ablations. A run with strong trace scores and invalid output
still fails normal parse/fidelity/structure/reward gates.

## Checkpoint boundary

Topology checkpoints are `grammar_diffusion` format v2. Runtime loading rejects
fixed-canvas v1 checkpoints and names the required command:

```bash
python -m scripts.migrate_checkpoint \
  --checkpoint path/to/fixed-canvas.pt \
  --output path/to/topology-v2.pt
```

Migration copies exact-shape tensors, initializes new topology heads/embeddings,
and writes `.migrate.json` with copied, skipped, and initialized keys. It is a
warm start only; the output requires topology training and honest evaluation.
There is no legacy fixed-canvas runtime alias.

## Experiment contract

X2-X8 are frozen fixed-canvas evidence and can only be reproduced from their
recorded source commit. X9-X15 are cumulative topology ablations:

1. X9 typed tree state and synchronous expansion;
2. X10 edit actions;
3. X11 structural embeddings;
4. X12 heterogeneous node noise;
5. X13 critic-guided accept/defer/contract;
6. X14 bounded active buffer plus global synchronization;
7. X15 full stack with curriculum and capacity.

Default screening is seeds 0/1/2, 80 CPU scratch steps, and median-by-experiment
successive halving on smoke, held-out, then adversarial. Halving retains at least
two experiment rows; survivors receive the full suite. `--confirm-steps 200`
restarts those two rows at the larger budget without halving. Full HF-context
training and bucket sync require a separate decision; this implementation cycle is
local scratch only.

Negative results complete an experiment only when the recipe, suite sizes,
AgentEvals/AgentV bundle, raw JSON, markdown headline, checkpoint disposition, and
unchanged gate result are all durable.

## Measured implementation smoke (2026-07-15 UTC)

Durable summary: [grammar-topology-smoke-results.json](grammar-topology-smoke-results.json).
This is a two-record CPU/scratch overfit diagnostic, not X-matrix or ship evidence.

| Revision | Steps / n | Parse / fidelity | Topology composite | Result |
| --- | ---: | ---: | ---: | --- |
| Concrete-type collapse | 200 / 2 | 0.0 / 0.0 | unavailable | Failed: training/inference mask-type mismatch; first temp AgentV bundle was not retained, so this run is diagnostic only |
| Generic expression mask | 200 / 2 | 0.5 / 0.5 | 0.4820 | Wiring improvement; still fails quality, no promotion |

The retained run used batch 2, learning rate `3e-3`, scratch context, 64 hidden
dimensions, one context layer, two denoiser layers, and honest request slot
contracts. AgentV executed 5 checks with 0 execution errors and 0 passes. The
checkpoint stayed under pytest temporary storage, was not synced, and is not a
reusable champion. The result validates that topology states can grow into a valid
program and that the new metrics persist; it does not validate generalization.

An initial X9 launch was canceled after 6/80 optimizer steps (last loss 27.5895)
before checkpoint or evaluation because the implementation had not yet been
committed and would have inherited the frozen baseline SHA. It is recorded in the
JSON as setup telemetry only, not matrix evidence. The measured matrix starts from
the committed implementation revision.
