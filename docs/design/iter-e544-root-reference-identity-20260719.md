# E544 — bounded root-reference identity

E544 adds a bounded multi-label target for the identities referenced by the
terminal structural root. The target is defined by generated-section indices,
not example literals, and masks tokenizer-tail classes beyond the sections
available in each record. Of 244 training records, 188 are covered; 42 of
those covered targets (22.34%) are strict subsets rather than the trivial
"reference every generated section" pattern.

The local scratch continuation `e544-e543-root-identity1-r2-24s` starts from
E543 and runs 24 CPU HF-context steps / 1,270 target tokens in 40.96 seconds
under `max_wall_minutes=3`. Identity positive recall rises from a 0.3056 mean
over steps 1–12 to 0.5729 over steps 13–24 while mean loss falls from 0.8193
to 0.7303. Exact-set accuracy remains weak at 0.0417 in the second half. The
checkpoint SHA is
`3b6e3c00666b8832187a489d6684ce909fff5b3ccaef57965f9cc1975474f20c`.
This was an explicit no-sync scratch diagnostic, not a full train or promotion.

Decoder telemetry invalidated the first implementation: additive inclusion
log-probabilities could demote all references below a non-reference candidate,
allowing identity to override arity. The accepted decoder instead permutes the
existing legal-reference scores by learned identity rank, assigns the maximum
existing reference score to the best unused reference, and leaves every
non-reference score unchanged. At operational weight 1, this preserves the
reference-group maximum exactly. The final trace records 11 changed identity
choices, all reference-to-reference.

The same-checkpoint, same-commit four-record OOD comparison changes only the
identity decode weight from 0 to 1. Meaningful-v1 rises from 0.00 to 0.25,
structural similarity from 0.1250 to 0.1688, component recall from 0.1458 to
0.2708, and AST node F1 from 0.1833 to 0.2833. Syntax stays 1.0, fidelity
0.4333, validity 0.6600, and reward 0.7370. Strict binding-aware meaning and
AST edge F1 remain 0.0; AgentV fails 0/1 without an execution error.

**Verdict:** retain bounded identity supervision and the rank-only decoder,
default-off. The diagnostic topology gain is causal and useful, but this small
OOD subset is not a ship evaluation and the checkpoint is not promotable. The
next lever should improve exact-set calibration or coverage-conditioned
identity learning without weakening any gate. Machine-readable evidence:
[JSON](iter-e544-root-reference-identity-20260719.json).
