# Autotrain c1722: decode wall-timeout on both fresh arms

**Verdict:** fixture/scratch measurement incomplete. Both training stages ran
fresh (no reuse — this loop-id's local campaign root has no prior state) and
wrote checkpoints, but evaluation decode on both the `control` and `canvas`
arms exceeded the cycle wall budget before either produced a scoreable
smoke result. No arm comparison or promotion is authorized.

## Result matrix

| Arm | Training | Smoke result | Gate / completion | Disposition |
| --- | --- | --- | --- | --- |
| control | fresh, 21 steps, 1,608,962 params, checkpoint written | No metrics | Evaluation wall timeout mid-decode | Frozen replay required |
| canvas | fresh, 21 steps, 1,608,962 params, checkpoint written | No metrics | Evaluation wall timeout mid-decode | Frozen replay required |

The training stage for both arms completed and persisted checkpoints under
`outputs/autoresearch/.../runs/<arm>/checkpoints/` (gitignored, local-only).
Evaluation on both arms was interrupted by the driver's cooperative wall-time
enforcement before any AgentV scoreboard was produced. The frozen manifest
digest for the pending retry is
`b1ec053df914d11a5648aa57ba0ea8ca5098a3cc8c399bc58fe872c5dc1ccd4f` (see the
[machine-readable record](autotrain-cycle-1722-decode-wall-timeout.json)).

## Diagnostic signal

Both arms' interrupts land in the same completion-kernel recursion family
documented across c1715–c1720, one frame deeper than c1720's fix touched:

```text
_compiler_ltr_decode_batch
  -> _greedy_ltr_decode_batch
  -> build_completion_forest
  -> GrammarCapabilityAdapterV1.completion_domain
  -> _openui_completion_domain
  -> session.terminal_witness
  -> completion_kernel._eval (recursed 12+ frames)
  -> advance_path -> advance
  -> semantic_state.advance -> dataclasses.replace
```

This container is a fresh CPU-only bootstrap (see
[c1721](autotrain-cycle-1721-fresh-container-bootstrap-screen.md)) whose raw
per-forward CPU throughput has not been benchmarked against the host that
produced c1715–c1720's timings; the c1720 uncached-fingerprint repair is
present (`model.twotower=v274`, this session's eval commit
`c387f3e57662198a68a89c978c6d4dceeb6533b3` descends from it) but did not
prevent this timeout, and the interrupted frame (`semantic_state.advance`)
differs from c1720's (`sorted(tokenizer.token_to_id.items())`). This is one
observed interrupt trace on a different frame than the prior fix targeted —
it is not yet a repeated (3x) blocker on the *same* signature, and per the
continuous-loop absolute law a single wall-timeout is a soft failure that
never stops the loop. No source repair is proposed from one sample; a repair
here would require counting exact-state evaluations and call frequency for
`semantic_state.advance` the way c1719's prefill probe did before touching
scheduling, which is out of scope for this cycle.

## Next-run priorities

1. Replay the identical frozen manifest
   (`b1ec053df914d11a5648aa57ba0ea8ca5098a3cc8c399bc58fe872c5dc1ccd4f`) per
   the pending `retry_measurement` action; only a complete fresh scoreboard
   authorizes a comparison.
2. If the same `semantic_state.advance` frame recurs on the identical arm
   signature two more times, escalate to a counted probe (à la c1719) before
   any scheduling or caching change — never guess a fix from one trace.
3. Keep ship gates on; no theorem optimum, checkpoint, or promotion evidence
   was produced, so no model-card change is required.

No component version bump — this cycle changed no versioned file.
