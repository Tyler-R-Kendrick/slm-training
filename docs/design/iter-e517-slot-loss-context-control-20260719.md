# E517 — slot-loss context control

E517 is matched to E515 except slot-component loss returns `4→1`. Focal gamma
remains zero and honest slot-contract context remains enabled during training.
The E396 parent, E500 primary corpus, 50% E357 replay, seed, learning rate, and
5,000-target-token budget are unchanged.

The CPU HF-context run stopped normally after 101 steps and 130.7 seconds,
inside `max_wall_minutes=3` and the external 170-second cap. Its serving
checkpoint SHA-256 is
`2b572a04256db14095e813e146079af9e6f6c948963d60f2bd669855e24b60e3`;
all checkpoint artifacts are uploaded and verified at
`hf://buckets/TKendrick/OpenUI/checkpoints/e517-e396-e500-replay050-slotrole1-context-r1-5k`.

E518 evaluates that exact checkpoint on the matched four-record OOD suite and
publishes AgentEvals plus pinned AgentV evidence.

| Arm | Slot loss | Training contract context | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | AgentV |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E510 leading policy | 1 | No | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 0/1 |
| E515 focal-zero | 4 | Yes | 0.25 | 0.6583 | 0.3213 | 0.2708 | 0.8270 | 0.4292 | 0.0625 | 0/1 |
| E517 context control | 1 | Yes | 0.00 | 0.4083 | 0.2250 | 0.2083 | 0.7445 | 0.2833 | 0.0625 | 0/1 |

## Decision

Reject E517. Lowering slot loss in the context-conditioned recipe makes every
headline metric worse, so loss scale and context conditioning interact.
Neither context-conditioned arm approaches E510, and strict binding-aware
meaning plus AgentV remain zero. Stop objective-scale tuning; remove or
redesign training-time contract context and improve role-label representation
in the corpus.

Exact recipe, provenance, metrics, and gate outcome:
[machine-readable record](iter-e517-slot-loss-context-control-20260719.json).
