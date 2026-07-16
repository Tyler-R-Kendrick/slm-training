# E125 DFA probe profile — 2026-07-15

E125 compares the existing copy-based DFA admission probes with the fallback
admission path on the E123 checkpoint.

| Variant | Seconds / generation | DFA syncs | Pick ms | Parseable |
| --- | ---: | ---: | ---: | --- |
| Copy probes enabled | 12.34 | 585 | 2,947 | No |
| Copy probes disabled | 11.86 | 574 | 2,723 | No |

The no-copy path is only about 4% faster for this one prompt and does not
produce legal output. The profile switch is retained, but the serving default
is unchanged. This is diagnostic-only evidence; the next decoder experiment
needs to reduce candidate work more substantially without bypassing legality.
