# Autotrain c1723: frozen-replay decode timeout, second consecutive

**Verdict:** fixture/scratch measurement incomplete for the second
consecutive cycle. This cycle is the automatic frozen replay of c1722's
manifest (`b1ec053d…`): both arms reused the c1722 `control`/`canvas`
checkpoints (`--checkpoint .../c2/.../checkpoints/last.pt`, no retraining),
but evaluation decode on both arms again exceeded the cooperative wall
budget before a scoreable smoke result. No arm comparison or promotion is
authorized.

## Result matrix

| Arm | Checkpoint source | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | c1722 control checkpoint (reused, no retrain) | No metrics | Evaluation wall timeout mid-decode | Frozen replay required |
| canvas | c1722 canvas checkpoint (reused, no retrain) | No metrics | Evaluation wall timeout mid-decode | Frozen replay required |

The new pending retry binds to frozen manifest
`dadf7da195133710c18a65eebb53381c126af2a36a482e7457c498371a20b0d3` (see the
[machine-readable record](autotrain-cycle-1723-frozen-replay-second-timeout.json)).

## Diagnostic signal

The canvas arm interrupted in the same recursive family as c1722 but one
call deeper — inside `completion_kernel.advance`'s engine fork:

```text
build_completion_forest -> session.terminal_witness
  -> completion_kernel._eval (recursed 7+ frames)
  -> advance_path -> advance
  -> engine.copy_control -> engine._copy -> engine._ip_control_copy
  -> copy.copy(parser_state.parse_conf)
```

This is the second consecutive cycle where both arms failed to complete
decode within the wall budget (`MAX_RUN_MINUTES=3` per stage, repo-wide,
non-negotiable — see `src/slm_training/levers.py`), each time interrupting
inside a different frame of the same `completion_kernel._eval` recursion
under `terminal_witness`. This container has 4 CPU cores and no GPU. Two
consecutive same-family timeouts (c1722, c1723) is not yet the three-strikes
threshold the continuous-loop absolute law requires before declaring a hard
block, and both are still soft failures that must not stop the loop — but it
is now a real pattern worth naming rather than a one-off: **honest smoke
evaluation of this fixture model has not completed in this container in two
of three cycles this session**, versus completing in the first cycle
(c1721). No repair is proposed yet — a third identical-signature timeout
would warrant the counted-probe escalation from c1719's playbook (exact
call counts per frame) rather than a guessed scheduling change.

## Next-run priorities

1. Replay the identical frozen manifest
   (`dadf7da195133710c18a65eebb53381c126af2a36a482e7457c498371a20b0d3`).
2. If the next replay also times out in the same `completion_kernel`
   family, treat it as the three-strikes hard block, stop guessing, and run
   a counted per-frame probe (à la c1719) before any scheduling change —
   never spend a repair attempt without call-count evidence.
3. No theorem optimum, checkpoint promotion, or model-card change from this
   cycle.

No component version bump — this cycle changed no versioned file.
