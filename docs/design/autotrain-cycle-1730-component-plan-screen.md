# Autotrain c1730: size-matched component-plan screen

**Verdict:** both CPU scratch arms completed all three smoke records with valid,
fail-closed grammar output and no runtime or AgentV errors. Component-plan loss
weight `1.0` changed neither decoded quality nor any structural metric. The
candidate was exactly size-matched and reduced fixture decode p50 by 13.23%, so
the controller queued it for confirmation as an efficiency-only signal. It is
not a quality win, a reusable checkpoint, or ship evidence.

## Result matrix

| Arm | Records | Trainable params | Parse | Binder F1 | Meaningful | Structure | Component recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control, component-plan head prebuilt | 3/3 | 1,755,764 | 1.0 | .6333 | .3333 | .1742 | .25 | .2619 / 0.0 | 1,798.92 ms | Complete fixture control; ship gates fail |
| component-plan loss `1.0` | 3/3 | 1,755,764 | 1.0 | .6333 | .3333 | .1742 | .25 | .2619 / 0.0 | 1,560.93 ms | Quality null; efficiency signal queued for confirmation |

The primary structural-similarity delta is exactly zero. Placeholder fidelity
(.5278), reward (.7653), AST scores, and every decoded structural signal are
also identical. The candidate's training loss is worse (25.0565 vs 20.5741)
and its training wall is 2.78 times the control (8.343 s vs 3.001 s). This is a
three-document fixture screen, not evidence that component-plan supervision
improves OpenUI programs.

## Runtime and formal signal matrix

| Signal | Control | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| decode p50 | 1,798.92 ms | 1,560.93 ms | Candidate is 13.23% faster on this fixture |
| decode total | 5,003.991 ms | 4,785.292 ms | Small aggregate efficiency signal |
| compiler | 3,475.522 ms | 3,334.245 ms | Most of the observed difference |
| backbone | 1,235.338 ms | 1,170.666 ms | Same 14-forward schedule |
| unique completion states | 31,142 | 31,142 | Identical grammar-authority work |
| witness states / parser forks | 906 / 32,174 | 906 / 32,174 | Identical deterministic completion path |
| AgentV execution errors / decode timeouts | 0 / 0 | 0 / 0 | Measurement complete |
| LeverProof band | not bound | not bound | Screening role does not authorize promotion; formal preflight remains mandatory for promotion |

The terminal matrix currently renders an em dash for the last row, which is
ambiguous between "not applicable to screening" and "missing promotion
evidence." The next harness pass should make that state explicit and keep a
missing promotion certificate fail-closed.

The honest gate fails smoke `n=3`, meaningful-program rate, structure,
component recall, AST BEq, and canonical BEq. Held-out, adversarial, OOD, and
full RICO suites were not run. Parse, placeholder fidelity, and reward pass the
fixture thresholds, but those partial clears do not change the blocked ship
state.

## Next-run priorities

1. Confirm the component-plan efficiency signal at a new seed with the same
   1,755,764-parameter geometry; reject it if quality changes or the latency
   gain does not repeat.
2. Keep structural similarity as the declared primary. Do not relabel the
   latency signal as a model-quality gain.
3. Render screening formal status as explicit `not_applicable`; retain
   `missing` as a fail-closed state for confirmation/promotion campaigns.
4. If confirmation is null, rotate to the size-matched component-inventory
   arm. Preserve parse, meaning, binder F1, component recall, AST node/edge F1,
   and completed latency in the terminal matrix.
5. Any promotion attempt must bind the locked metric expectations digest, pass
   the Mathlib-free LeverProof preflight, export/replay a v2 certificate, and
   retain AgentEvals and honest ship gates as verdict authority.

Machine-readable evidence is in
[`autotrain-cycle-1730-component-plan-screen.json`](autotrain-cycle-1730-component-plan-screen.json).
