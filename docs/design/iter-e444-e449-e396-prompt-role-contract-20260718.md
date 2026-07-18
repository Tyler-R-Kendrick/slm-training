# E444–E449 E396 prompt-role contract — 2026-07-18

E444–E449 repair E396's only full-RICO placeholder-fidelity miss without
reading hidden gold structure. The failing prompt explicitly requests two form
controls with role `datepicker`; the gold program contains two `DatePicker`
components and an explicitly empty placeholder contract.

All six one-row diagnostics used the unchanged E396 checkpoint SHA, CPU, local
HF context, 320-token grammar LTR, automatic content floor, component-plan
weight 2, slot-component weight 8, honest constrained slot contracts, eight
generation steps, and three attempts. Each process was externally capped at
290 seconds with a ten-second forced kill. Every run completed normally; no
timeout, fallback, or execution error is evidence here.

| ID | Change localized | Meaningful | Fidelity | Structure | Type recall | Reward | p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E444 | Initial prompt-role implementation | 0.0 | 0.0 | 0.4000 | 0.0 | 0.000 | 2.87s |
| E445 | Constrain initial choice state | 1.0 | 0.0 | 0.6917 | 1.0 | 0.799 | 2.94s |
| E446 | Preserve explicit empty slot contract | 0.0 | 1.0 | 0.3833 | 0.0 | 0.000 | 9.77s |
| E447 | Count prompt roles in density floor | 0.0 | 1.0 | 1.0000 | 0.0 | 0.000 | 3.17s |
| E448 | Allow zero-slot gold in meaningful gate | 1.0 | 1.0 | 1.0000 | 1.0 | 0.000 | 3.39s |
| E449 | Allow zero-slot gold in reward grammar | 1.0 | 1.0 | 1.0000 | 1.0 | 0.961 | 3.16s |

The sequence identifies one decoder omission and two evaluator assumptions:

1. E444 did not constrain the undecided initial choice state, so it reproduced
   E441's `TextContent(":hero.title")`.
2. E445 forced the requested type, but the heuristic fallback slot polluted a
   genuinely empty contract and limited generation to one control.
3. E446 preserved the empty contract. Direct token inspection showed that two
   `DatePicker` declarations were generated, but the zero semantic-density
   floor allowed canonicalization to collapse the root to `Stack([])`.
4. E447 counted prompt-declared instances in the density floor and generated
   the structurally exact two-control graph. The meaningful evaluator still
   rejected every placeholder-free program.
5. E448 made meaningfulness gold-aware: placeholder-free output is accepted
   only when the supplied gold contract is also empty. Its structure, type
   recall, and placeholder metrics all reached 1.0.
6. E449 applied the same narrow exception to the structure-only reward grammar.
   Generic and RL scoring without gold remains strict.

E449's output is:

```openui
root = Stack([v0, v1])
v0 = DatePicker("item")
v1 = DatePicker("ntnn")
```

AST node F1, AST edge F1, tree-edit similarity, and reference-graph exactness
are all 1.0. Canonical exact match remains zero because literal names differ
from gold; this is not hidden as an equivalence claim.

Each diagnostic emitted AgentEvals JSONL and an AgentV result bundle. AgentV is
0/5 with zero execution errors for each because a one-row diagnostic cannot
supply the four bounded suites or the full 1500-row RICO gate.

**Verdict:** prompt-declared roles are a valid, inference-visible constraint
for recovering zero-slot controls. The single known fidelity miss is repaired,
but this one-row result is not a ship or promotion claim. Run a matched
contiguous RICO shard next to measure regressions before any full-suite
expansion.
