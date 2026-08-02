# Autotrain c1800: mixed masking is null

**Verdict:** reject mixed random/structured corruption at this recipe. Both
size-matched arms completed 20 CPU scratch steps and the full 3-record smoke
screen. Structure, binder-reference F1, fidelity, recall, meaningful-program
rate, and reward are exactly tied; mixed masking is slower and has worse loss.

| Arm | Params | Loss | Structure | Binder F1 | Fidelity | p50 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| random-mask control | 1,608,962 | 16.90084 | 0.135267 | 0.63333 | 0.52778 | 982.53 | retain baseline only |
| mixed mask | 1,608,962 | 22.29659 | 0.135267 | 0.63333 | 0.52778 | 1112.15 | reject approach |

Both arms have parse rate 1.0, meaningful-program rate 0.33333, component
recall 0.16667, reward 0.76533, three completed documents, and zero decode
timeouts. AgentV completed both bundles without execution errors. The primary
delta is exactly zero while candidate p50 regresses 129.62 ms. This is fixture
evidence only; honest ship gates fail.

The checkpoints are local scratch artifacts under the c1800 campaign, with
explicit no-sync policy. Control SHA-256 is `434e50e8...42727`; candidate
SHA-256 is `6b80bd63...832f9`. Neither is reusable, promotable, synced, or ship
evidence. Lean is `not_applicable:screening`; no confirmation or promotion
preflight was authorized.

The null exhausted the registered quality-arm bank. Campaign harness v100 adds
the next size-matched hypothesis: `symbol_boundary_loss_weight=1` versus zero.
This existing loss reweights opaque-symbol tokens and their immediate
boundaries through the same output logits, so it adds no parameters and cannot
alter deterministic decode legality. The same repair also makes campaign-lock
control fingerprints explicitly reset slot augmentation, mask pattern, and
boundary weight, preventing a candidate lock from describing its treatment as
the negative control. Train harness v32 records the boundary weight; TwoTower
v295 exposes it through the canonical CLI/compiler.

Next priority: test symbol-boundary supervision at identical size, seed policy,
and smoke evaluation. If it is null or harmful, inspect per-token boundary loss
coverage before adding another objective; do not increase model size or weaken
gates.

Machine evidence:
[`autotrain-cycle-1800-mixed-mask-null.json`](autotrain-cycle-1800-mixed-mask-null.json).
