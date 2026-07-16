# E84 grammar LTR primary with visible contracts — 2026-07-15

E84 retrained the visible-contract corpus with grammar LTR primary enabled and
the learned fast path disabled. The bounded smoke matrix probe (n=1, 128
steps) scored parse 1.0, exact placeholder fidelity 1.0, structural
similarity 0.65, and reward 0.997.

Inspection of the emitted prediction shows that this result is produced by
the existing certified template fallback after learned decoding fails, not by
successful learned structural generation. This motivated explicit decode
telemetry for `template_fallback_count` and `template_fastpath_count`.

Decision: reject E84 as learned-model evidence; retain the telemetry change and
continue separating certified fallback quality from model quality.

This is bounded scratch evidence, not a ship claim.
