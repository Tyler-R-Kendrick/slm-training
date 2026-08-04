# Autotrain c1775: component-edge is quality-null

**Verdict:** reject. The size-matched component-edge arm ties its control on
every measured smoke quality metric. Its 3.88% MPR/ms gain is below the
preregistered 5% minimum effect, so the latency difference is not a positive
result.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| component-edge (train/decode weights 1) | 1,766,987; 22 steps; loss 20.11267; 2.67 s | 3/3 complete; parse 1; meaning .3333; structure .17417; binder F1 .6333; fidelity .5278; reward .76533; p50 1,052.15 ms | null; retire |
| matched control (weights 0) | 1,766,987; 22 steps; loss 18.56901; 2.73 s | 3/3 complete; parse 1; meaning .3333; structure .17417; binder F1 .6333; fidelity .5278; reward .76533; p50 1,093.01 ms | baseline |

Both arms completed all records and emitted canonical AgentV bundles with zero
execution errors. The candidate's structure, meaning, binder, fidelity, and
reward values are exact ties, while its training loss is worse. The observed
latency movement is fixture-only and below the minimum-effect policy; it cannot
support promotion or a ship claim.

These are local CPU scratch checkpoints with explicit no-sync policy. The model
card and README record them for provenance, but they are rejected and must not
be reused, promoted, synced, or shipped. Lean is
`not_applicable:screening`: no champion exists, while formal preflight remains
mandatory for promotion. The next ranked hypothesis is the distinct,
size-matched `component-inventory` arm.

Machine evidence:
[`autotrain-cycle-1775-component-edge-null.json`](autotrain-cycle-1775-component-edge-null.json).
