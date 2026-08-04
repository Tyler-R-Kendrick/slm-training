# Autotrain c1779: binder topology is a quality null

**Verdict:** reject. Binder-topology supervision and its size-matched control
tie exactly on every smoke quality metric, while the candidate is 3.80% slower
at p50 and takes 2.37x as long to train.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| binder topology | 2,137,346; 20 steps; loss 14.68723; 6.26 s | n=3; parse 1; meaning 0; structure .11750; binder .8222; fidelity .7222; reward .31233; p50 1,480.73 ms | reject |
| matched control | 2,137,346; 20 steps; loss 14.68723; 2.65 s | identical quality; p50 1,426.49 ms | baseline |

Both AgentV bundles are complete with zero execution errors. Gates fail, and
this local CPU fixture is not ship evidence. Both explicit no-sync scratch
checkpoints are provenance-only and must not be reused, promoted, synced, or
shipped. Lean is `not_applicable:screening`: no champion reached confirmation,
so the fail-closed promotion proof preflight did not run.

The completed binder-topology family is exhausted. The next ranked distinct,
size-matched quality hypothesis is `component-structure`; the matched control
remains mandatory.

Machine evidence:
[`autotrain-cycle-1779-binder-topology-null.json`](autotrain-cycle-1779-binder-topology-null.json).
