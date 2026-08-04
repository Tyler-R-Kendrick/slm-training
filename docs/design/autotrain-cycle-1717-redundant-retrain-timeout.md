# Autotrain c1717: redundant retrain timeout

**Verdict:** fixture/scratch measurement incomplete. The matched control completed
honest smoke evaluation and failed its ship gates. The batch-size-1 candidate
exhausted its bounded stage during training and never reached evaluation. It wrote no
checkpoint, scoreboard, AgentV bundle, or scoreable metric. No quality or latency
comparison is authorized.

## Result matrix

| Arm | Recipe | Params | Smoke result | Gate / completion | Disposition |
| --- | --- | ---: | --- | --- | --- |
| control | CPU, 80 steps, batch 2, seed 101710 | 1,608,962 | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 3,954.45 ms | Complete AgentV run; honest gates fail on evidence volume, missing ship suites, and quality thresholds | Model evidence only; not ship |
| batch1 | CPU, 80 steps, batch 1, seed 101710 | size-matched declaration | No metrics | Interrupted in training backward; no checkpoint; incomplete | Frozen replay required |

The handoff remains `inconclusive`, `ship_state=blocked`, and binds its retry to
manifest `779b44e3c22b7ad840ad5e73212e3a92f63e107118b8fa0acab0c8e5b7999113`.

## Diagnostic signal and repair

c1715 and c1716 had already completed the identical deterministic batch1 train and
produced the same checkpoint SHA-256. c1717 nevertheless restarted both train stages.
Under transient CPU contention, the candidate consumed its stage in `loss.backward()`
and never exercised the merged decode repair.

The supervisor repair reuses a completed train stage only when all of these checks
pass closed:

1. the current locked manifest replays an ordered, hash-verified manifest lineage;
2. the lineage terminates at a clean source manifest matching the source run;
3. `train_summary.json` proves declared-step completion and exact steps, batch, seed,
   and learning-rate parity with the frozen command;
4. the checkpoint resolves inside that run, all required tokenizer/meta sidecars
   exist, and checkpoint plus summary digests are recorded;
5. evaluation uses the explicit historical checkpoint but writes into the successor
   run namespace; the execution plan and outcome telemetry record that training was
   reused and not executed.

Any missing or mismatched proof aborts reuse. This does not reuse evaluation results,
change gates, widen grammar authority, or bypass Lean promotion preflight. The next
authoritative test is the identical replay after merge.

## Checkpoint and provenance

| Arm | Local checkpoint SHA-256 | Status |
| --- | --- | --- |
| control | `e38a18adafcf2812e30f9f4ef748962b43834dc08fe57a99b43851b5fa7c54ee` | no-sync fixture; reject |
| batch1 | none | train incomplete; reject |

Train/eval commit: `f9da6539c42ceba43b803f371f9dc313e11a7dbe`
(`model.twotower=v271`, `harness.model_build.eval=v71`). Stage reuse is
`harness.autoresearch.experiment_campaign=v43`. Machine-readable evidence is in
[`autotrain-cycle-1717-redundant-retrain-timeout.json`](autotrain-cycle-1717-redundant-retrain-timeout.json).

## Next-run priorities

1. Replay the exact frozen control and batch1 manifests using only lineage-verified
   completed train stages; preserve evaluation recipe and wall limits.
2. Require both fresh authoritative scoreboards before comparing the arms.
3. If candidate evaluation still times out, use its new interrupt trace as the next
   exact harness signal; do not alter grammar, proof budget, or deadline.
4. Keep Lean formal preflight on promotion cadence. c1717 supplies no optimum or
   promotion evidence.
