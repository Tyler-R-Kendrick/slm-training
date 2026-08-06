# Continuous autotrain: 2026-08-05 (scheduled session `ttvqzi`) cycle 1 — control arm hits wall cap after eval completes, measurement incomplete

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `bdf143cd` (`origin/main` tip; clean checkout, fresh
`.venv-autotrain` — Python 3.12, `torch==2.5.1+cu124`, `pip install -e
".[dev]"` — plus the `src/apps/openui_bridge` and `src/apps/design_md_bridge`
JS grammar bridges installed per the continuous-mode prerequisite)
**Predecessor:** none (first cycle of a new scheduled session)

**Verdict:** non-positive / measurement incomplete. Command:

```
python -m scripts.run_autotrain_continuous --loop-id continuous-openui-local \
  --supervised --max-cycles 1 --train-version wf_smoke_v2 --steps 20
```

## Results

| Arm | exit | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 124 (timeout) | 1.0 | 0.0 | 0.0575 | 0.6333 | 4868.02 | no scoreboard (measurement incomplete) |
| bounds | 0 | 1.0 | 0.0 | 0.0575 | 0.6333 | 5138.03 | fail (gate reject, `n=3` fixture) |

The **control** arm's own `eval_smoke.json` shows the AgentEvals decode pass
actually finished (`completed_document_n=3`, `incomplete_document_n=0`,
`evaluation_wall_seconds=43.651774`) — but the process was killed by the
repo-wide `MAX_RUN_MINUTES=3` wall cap before `scoreboard.json` was written,
so there is no honest ship-gate verdict for control. The **bounds** arm
completed cleanly (`exit=0`) but fails every quality-threshold and
evidence-volume gate at fixture `n=3`, as expected for a 20-step/`n=3` smoke
screening cycle.

Because control never produced a scoreboard, the primary metric comparison
(`smoke.structural_similarity`, control vs. bounds, both `0.0575`) is not a
valid measurement — it is comparing bounds against its own metrics.jsonl
number, not against a completed, gated control. Per SDLC Phase A this is
`measurement_incomplete`, not a scoreable tie.

## SDLC Phase A

**Non-positive** (`measurement_incomplete:control:missing_scoreboard`,
`fixture_insufficient_n:bounds`, `fixture_insufficient_n_alone`). No stacked
PR layer for this cycle — this session's setup work (fresh venv + JS bridge
install, docs) stays as local commits on `claude/great-dirac-ttvqzi` per
`sdlc` autotrain-iteration-delivery.

## Next priorities

1. (rank 1, confidence 0.95) Replay the identical frozen control/bounds arm
   pair (`retry_measurement`, `frozen_manifest_sha256=1ada6cf3…`) before
   testing any new hypothesis — the driver's typed handoff queues this
   automatically for cycle 2.

Machine evidence:
[`continuous-openui-local-ttvqzi-c1-results.json`](continuous-openui-local-ttvqzi-c1-results.json).
