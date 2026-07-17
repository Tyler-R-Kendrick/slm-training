# E286 — batched exact-signature profiler rejected

Date: 2026-07-17
Status: **stopped; implementation rejected; invalid evidence**

E286 replaced separate reverse passes for grouped preference objectives with
chunked PyTorch batched vector-Jacobian products. The intended semantic contract
was unchanged: frozen E228 checkpoint, full committed E283 corpus (372 events),
`ftpo_set`, legal-token probability space, unit-normalized gradients, and the
same decision-kind/decision-signature strata as the committed E284 control.

## Bounded reproduction

The batch-16 implementation first attempted to reproduce E284 before any
exact-signature claim:

- device: CPU, 12 Torch threads;
- wall budget: five minutes, enforced by the process envelope;
- elapsed: 283.62 seconds;
- outcome: killed before completion;
- output report: none;
- training/checkpoint mutation: none.

Because no report was written, there are no metrics to compare with E284. The
attempt is invalid evidence. The batched implementation was removed rather than
retained on the strength of a toy derivative-equivalence test.

## Decision

Do not spend another bounded iteration tuning VJP batch size or reducing the
corpus. E285 remains an unresolved diagnostic, but this implementation cannot
service it under the experiment runtime policy. Continue with a matched,
five-minute-capable training lever. The registered B3 lexer/choice capacity
arms hold width, depth, target-token budget, steps, seed, and decode recipe
constant, directly preventing a larger training recipe from buying an
artificial comparison win.

Machine-readable record:
[quality-matrix-v10-e286-batched-profile-rejected-results.json](quality-matrix-v10-e286-batched-profile-rejected-results.json).
