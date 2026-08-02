# Autotrain 1pse54 c3: frozen replay confirms the AgentV fix, grammar_completion_bounds is quality-null

**Verdict:** the AgentV NODE_OPTIONS fix (commit `46fecfc`) is confirmed as a
genuine harness unblock: this cycle replays the exact frozen `c2`
control/`grammar_completion_bounds` arms, and both now complete evaluation
end to end with zero AgentEvals execution errors. The `grammar_completion_bounds`
treatment itself is quality-null against its size-matched control.

| Arm | Params | Parse | Meaningful | Structure | Binder F1 | AST/canonical BEQ | p50 | Decision |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- |
| control (bounds off) | 1,608,962 | 1.0 | 0 | 0.0575 | 0.6333 | 0 / 0 | 1,340.24 ms | complete gate rejection |
| bounds (`grammar_completion_bounds` on) | 1,608,962 | 1.0 | 0 | 0.0575 | 0.6333 | 0 / 0 | 1,494.47 ms | complete gate rejection |

Both arms are exact quality ties (identical checkpoints reused from the c2
frozen replay — no retraining occurred). `grammar_completion_bounds` costs
11.5% latency (1,340.24 → 1,494.47 ms) with zero quality change, so it is
rejected outright, not merely below an efficiency floor. Smoke `n=3` and every
non-smoke suite is missing, so this is fixture wiring evidence only — not a
ship claim.

## Provenance

No retraining occurred in this cycle: both arms reuse the exact c2
checkpoint bytes under the locked c2 frozen manifest
(`7ab08b83e6078e94edbe3605149524836350ab9aca008a40e8a271f67a586598`).

| Arm | Reused manifest SHA-256 | Reused checkpoint SHA-256 |
| --- | --- | --- |
| control | `c0286c22ef0ebb642c0d4d513886accc206468631537fd791d53fff9d919b1e0` | `808ee893a079a1e6cc7e2ac5d362ac73de39a52cc446bef328b937f1ef2efc36` |
| bounds | `7ab08b83e6078e94edbe3605149524836350ab9aca008a40e8a271f67a586598` | `094287fbbb61daf17d45316e362316db239f75c23e720f9866fba6da485e147c` |

These are the identical checkpoints documented in
[`autotrain-cycle-1pse54-c2-agentv-node-options-block.md`](autotrain-cycle-1pse54-c2-agentv-node-options-block.md);
full digest and path detail is in the `frozen_replay_provenance` block of
[`autotrain-cycle-1pse54-c3-grammar-completion-bounds-replay.json`](autotrain-cycle-1pse54-c3-grammar-completion-bounds-replay.json).

## Harness confirmation

This is the direct replay-proof requested by the c2 documentation
([`autotrain-cycle-1pse54-c2-agentv-node-options-block.md`](autotrain-cycle-1pse54-c2-agentv-node-options-block.md)):
the identical frozen arm that previously hard-failed in
`publish_agentv_evaluation` with `node: --import tsx is not allowed in
NODE_OPTIONS` now runs to a complete scoreboard. This satisfies the
"executable unblocking" bar for a positive SDLC delivery result on the
harness fix itself, independent of the model arm's quality-null outcome.

## Next step

Rotate to the distinct size-matched `component-plan` hypothesis (already
queued by the driver's speculative priorities) rather than iterating further
on `grammar_completion_bounds`.

Both checkpoints are local, explicit no-sync diagnostics reused unchanged
from c2. Neither is reusable, promoted, or ship evidence. Lean is
`not_applicable:no_champion`.

Machine evidence:
[`autotrain-cycle-1pse54-c3-grammar-completion-bounds-replay.json`](autotrain-cycle-1pse54-c3-grammar-completion-bounds-replay.json).
