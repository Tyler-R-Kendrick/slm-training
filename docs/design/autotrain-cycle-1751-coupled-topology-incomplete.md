# Autotrain c1751: coupled topology harness failure

**Verdict:** incomplete infrastructure evidence, not a model result. The matched
control trained and wrote a local checkpoint, but its evaluation process exceeded
the stage wall. The candidate failed before training because the frozen lexer
recipe enabled `binder_topology_decode_weight=1.0` without the required compiler
decode mode. No arm comparison exists and neither outcome may be scored.

## Result matrix

| Arm | Params | Train | Eval | Scoreable metrics | Disposition |
| --- | ---: | --- | --- | --- | --- |
| matched control | 2,137,346 | 24/24 steps; loss 12.2308; 2.811 s | process exceeded stage wall after writing partial artifacts | none | incomplete; checkpoint provenance only |
| binder topology loss `0.25` + decode `1.0` | same intended head capacity | rejected before step 1 | not run | none | invalid preregistered capability combination |

The control's interrupted files contain three finalized rows, but repository
policy forbids using a timed-out process as evidence. The candidate has no
checkpoint, evaluation, or AgentV bundle. The local control checkpoint is not
reusable, promotable, synced, or ship evidence.

## Harness signal and repair

The model capability gate behaved correctly and must stay fail-closed. Compiler-
path ranking levers require `compiler_decode_mode=tree`; the continuous matrix
did not bind that companion knob or match it on the control. The supervisor also
classified the deterministic failed experiment as a generic retry, which would
replay an impossible frozen manifest.

The canonical repair therefore:

1. binds `compiler_decode_mode=tree` into every structural train/decode arm and
   the matched control, and retains it through confirmation and promotion;
2. classifies a failed experiment outcome as `harness_failure` immediately so
   the handoff requires canonical repair instead of spending an identical retry;
3. leaves the frozen c1751 manifest immutable and advances only with a new
   preregistered successor after the repair receipt; and
4. preserves I6: compiler mode and auxiliary weights rank only legal candidates.

No empirical optimum band or champion exists. Lean status is
`not_applicable:screening`; actual promotion remains blocked on the mathlib-free
LeverProof preflight and replayed metric certificate. RL remains locked.

Machine-readable evidence is in
[`autotrain-cycle-1751-coupled-topology-incomplete.json`](autotrain-cycle-1751-coupled-topology-incomplete.json).
