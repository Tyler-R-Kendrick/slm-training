# E683 — independent v137 Held-out confirmation

Date: 2026-07-21
Status: completed negative confirmation; not ship

E683 independently replays retained v137 on the full five-record Held-out
suite. The capped CPU run completed without timeout or fallback and emitted
AgentEvals JSONL plus an AgentV SDK bundle.

| Held-out `n=5` | E683 v137 |
| --- | ---: |
| syntax / meaningful v1 | 1.0000 / 0.8000 |
| strict v2.4.0 / coverage | 0.4000 / 0.8000 |
| fidelity / validity | 0.7333 / 0.8400 |
| structure / component recall | 0.4933 / 0.5333 |
| reward | 0.8702 |
| AST node / edge F1 | 0.5773 / 0.5203 |
| latency p50 / p95 | 3344.37 / 20817.96 ms |
| timeout / fallback | 0 / 0 |
| AgentV | 0/1 |

The login record is structurally exact and the dual-card record clears strict
v2. Three failures remain:

- The form prediction repeats `hint.title` in incompatible Form/FormControl
  positions, triggering role mismatch and placeholder spam.
- The hyphenated/plural “two-tab” prompt produces unknown contract coverage,
  collapses to Image, and omits five visible slots.
- The settings prediction emits only Slider, omits the switch/description
  roles, and places `notify` in `Slider.variant`.

Retain v137 for its independently verified Smoke result, but reject the
hypothesis that it generalizes across Held-out. The next lever should isolate
the hyphenated/plural component-recognition gap before mixing in the separate
form and settings failures. This is not multi-suite ship evidence. No
checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e683-positional-role-heldout-20260721.json).
