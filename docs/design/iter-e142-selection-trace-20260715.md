# E142 constrained-decoder selection trace — 2026-07-15

E142 adds a bounded 64-event selection trace to `DecodeStats` and runs one
normalized smoke record against the E135 checkpoint.

The first observable divergence occurs at position 5:

```text
prefix:      root=CheckBoxItem(
chosen:      ")\\ncta = Button("
model argmax: =
forced:      false
```

The chosen token creates a malformed multi-line string-like continuation.
Later choices preserve that malformed prefix, so constrained decoding never
recovers and the final program does not parse. Feedback was parse `0.0`, reward
`0.0`, placeholder validity `0.40`, structural similarity `0.0538`, with no
timeout for this one-record diagnostic.

The trace also exposed a telemetry limitation: `legal_candidates=0` was
reported for every choice because the picker returns early on legal argmax,
forced, or singleton decisions without updating its last-candidate counter.
That value must not be interpreted as proof that no legal token existed.

Next harness change: classify the picker return path and make candidate-count
telemetry truthful before using this trace to evaluate the singleton-token
bypass hypothesis.
