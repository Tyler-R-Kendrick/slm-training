# E1090-E1091: root-tail-pair fresh train and matched rejection

E1090 is the valid fresh 395-step training comparison for E1088's immutable
532-row snapshot. It preserves E1085's CPU scratch TwoTower/lexer/tree recipe,
seed, honest slot contract, and four unit-weight structural objectives; only
the train snapshot changes.

The run completed exactly 395/395 steps through six explicitly chained full
states (44, 101, 183, 259, 380, and 395 cumulative steps), consuming 487.69
cumulative seconds. Its final loss is 8.7755; this scalar is not a quality or
ship metric. The local checkpoint is
`outputs/runs/e1090_v273_root_tail_pair_valid/checkpoints/last.pt` with SHA
`70adedee...83576`. It is unsynced and not a parent candidate.

E1091 evaluates exactly the held Settings row and strict compiler policy used
for E1087, with root-order decode weight 1. It regresses from E1087's parse
1.0, fidelity 0.3333, structure 0.06, reward 0.707, and no timeout to an empty
prediction after the 12.01-second decode timeout: parse/fidelity/structure/
reward all 0. AgentEvals JSONL and the pinned `@agentv/core` bundle are under
`outputs/runs/e1091_v273_root_tail_pair_settings_decode1/agentv/`.

Reject E1090 and E1088 as a positive intervention; do not promote, serve,
sync, or use this checkpoint as a parent. E1089's incomplete/restarted
artifacts remain invalid and excluded.
