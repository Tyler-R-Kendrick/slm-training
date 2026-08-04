# Autotrain c1765: numeric literal-close margin regresses quality

**Verdict:** reject the preregistered `literal-margin` arm. The direct
grammar-derived objective was active and learned its narrow training target,
but it did not improve OpenUI programs. Both CPU scratch arms ran 20 requested
steps on `wf_smoke_v2` with seed 101765, batch size 2, and exactly 1,608,962
trainable parameters. The treatment alone enabled compiler alignment with
`loss_weight=1`, `margin=1`, stratified selection, and the `literal-close`
kind filter.

| Arm | Params | Train wall | Last weighted loss | Smoke n=3 | AgentV / gates | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| literal-margin | 1,608,962 | 6.623 s | 12.7394 | parse 1; meaning 0; structure .11500; binder F1 0; component recall 0; p50 1,022.87 ms | bundle complete; no execution errors; fail | rejected quality regression |
| matched control | 1,608,962 | 2.703 s | 14.1693 | parse 1; meaning .3333; structure .14777; binder F1 .5; component recall .1667; p50 1,134.45 ms | bundle complete; no execution errors; fail | complete gate rejection |

The mechanism was not inert. Three of 21 emitted training-step records carried
one eligible numeric-close row each. Their margin-violation observations were
`1.0, 1.0, 0.0`, and the final active-row alignment loss fell to `.07367`.
The treatment checkpoint SHA differs from control. This establishes narrow
objective activation, but not useful program learning: meaningful-program rate,
binder/reference F1, component recall, placeholder fidelity, and reward all
fell to zero, while structural similarity fell 22.17%. The auxiliary work also
made training 2.45x slower.

Candidate smoke p50 was 9.84% lower, but `n=3` is fixture evidence and the
quality non-regression failures forbid an efficiency-only promotion. Raw final
losses are not directly comparable because the treatment optimizes an added
objective. Both arms completed all three selected documents with zero decode
timeouts and no fallback, so this result is attributable to the model arm, not
an incomplete evaluation.

This rejects a strong unit-weight numeric-close margin as a quality lever; it
does not reject the I6 constrained-decoding goal or deterministic completion.
The next run should not repeat this arm. The current handoff proposes the
distinct size-matched `component-plan` arm, but that family already has repeated
quality-null fixture results. Before spending another cycle there, prioritize
the harness improvements that make evaluation completeness fail closed,
content-bind promotion replicates, preserve AB/BA arm ordering, and budget Lean
formal preflight only from time remaining after both measured arms. Then choose
a hypothesis with a direct held-out transition or program-level causal metric,
not merely another auxiliary loss.

Both checkpoints are local, explicit no-sync diagnostics. Neither is reusable,
promoted, or ship evidence. Lean is `not_applicable:no_champion`; promotion
formal proofs remain mandatory when a champion exists.

Machine evidence:
[`autotrain-cycle-1765-literal-margin-regression.json`](autotrain-cycle-1765-literal-margin-regression.json).
