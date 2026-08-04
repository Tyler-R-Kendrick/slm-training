# Autotrain c1729: binder-topology null and capacity audit

**Verdict:** both CPU scratch arms completed all three smoke records with valid,
fail-closed grammar output and no runtime or AgentV errors. Binder-topology loss
weight `0.25` changed neither decoded output nor any quality metric. The arm is
rejected. It is not valid size-matched evidence: the treatment instantiated a
528,384-parameter auxiliary head that the control did not carry.

## Result matrix

| Arm | Records | Trainable params | Parse | Binder F1 | Meaningful | Structure | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1,608,962 | 1.0 | 0.0 | 0.0 | 0.0575 | 0.0 / 0.0 | 1,520.36 ms | Complete fixture control; ship gates fail |
| binder topology `0.25` | 3/3 | 2,137,346 | 1.0 | 0.0 | 0.0 | 0.0575 | 0.0 / 0.0 | 1,532.49 ms | Complete null; capacity-confounded and rejected |

Both arms emitted the same seven-token `Input`-root program for all three records.
The candidate added 528,384 trainable parameters (+32.84%), increased final loss
from 10.6642 to 10.7177, and was 1.008 times the control p50. The primary delta is
exactly zero. This is a three-document fixture diagnostic, not ship evidence.

## Runtime and signal matrix

| Signal | Control | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| training wall | 2.844 s | 7.911 s | Treatment costs 2.78x while training the extra head |
| decode total | 4,745.335 ms | 4,812.252 ms | No runtime win |
| compiler | 3,410.189 ms | 3,424.558 ms | Same certified search path |
| backbone | 1,104.372 ms | 1,123.596 ms | Noise-scale difference |
| neural forwards | 12 | 12 | Identical compute schedule |
| unique completion states | 31,483 | 31,483 | Identical grammar authority work |
| witness states / parser forks | 1,229 / 32,859 | 1,229 / 32,859 | Identical deterministic completion path |
| AgentV execution errors / decode timeouts | 0 / 0 | 0 / 0 | Measurement complete |

The smoke ship gate correctly fails `n=3`, meaningful-program, structure,
component recall, BEq, placeholder fidelity, and reward. Missing held-out,
adversarial, OOD, and full RICO suites keep ship state blocked.

## Harness feedback and repair

c1729 exposed three related supervisor defects:

1. training parameter counts existed in `train_summary.json` but were discarded
   before the outcome and terminal matrix, so the matrix printed `—`;
2. structural loss weights conditionally built auxiliary heads, making the new
   “size-matched” screening arms capacity-confounded; and
3. a completed null retained the already-executed arm as rank-one next-run advice.

The canonical harness now carries `track.trainable_params` into every outcome and
uses the train summary as a fail-closed fallback for Phase A and terminal display.
`structural_aux_head_profile` separates head construction from loss weight: the
zero-loss control prebuilds the same profile as the recommended treatment, so both
arms expose identical trainable capacity without changing decode weights. Confirm
and promotion cycles retain that profile. Post-outcome handoffs replace completed
null steering with a distinct quality hypothesis, and terminal status renders the
post-outcome handoff priorities instead of the preregistered pre-run recommendation.

No constraint, deterministic completion, output-contract, or ship gate is weakened.
Capacity growth still requires size parity within a causal pair and remains subject
to `EG_params` for comparisons against smaller champions.

## Next-run priorities

1. Run component-plan supervision with `component-plan` heads present in both the
   zero-loss control and treatment; verify equal trainable parameters in the terminal
   matrix before interpreting quality.
2. Treat the c1729 binder-topology arm as exhausted on this data/eval identity; do
   not rerun it merely because its preregistered priority was rank one.
3. If component-plan is also null, use component-inventory next; retain AST node,
   edge, component recall, meaningful-program rate, and latency as diagnostic axes.
4. Confirm any fixture win at a new seed before promotion. Promotion still requires
   the Mathlib-free LeverProof preflight, held-out primary effect, parameter charge,
   full AgentEvals evidence, and honest ship gates.
5. Increase training depth only as a separately preregistered, capacity-accounted
   experiment; c1729's 23-step outputs are syntactically valid but semantically empty.

Machine-readable evidence is in
[`autotrain-cycle-1729-binder-topology-capacity-audit.json`](autotrain-cycle-1729-binder-topology-capacity-audit.json).

