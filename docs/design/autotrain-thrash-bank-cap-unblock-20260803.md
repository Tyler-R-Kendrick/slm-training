# Thrash bank empty via causal CAP (2026-08-03)

## Symptom

Loop **BLOCKED** at c2260 with:

```text
registered screening arm bank exhausted; add a distinct preregistered quality objective…
```

Live ledger: multi-seed thrash close left **one** open arm (`literal-close`);
two same-tip confirm rejects burned **causal family CAP** (`_CAUSAL_FAMILY_ATTEMPT_CAP=2`)
so that arm was also skipped; no promote head → hard bank exhaust.

## Fixes

1. **Causal CAP relax:** if multi-seed-open thrash arms remain but CAP empties the
   bank, drop CAP skips (`THRASH_CAUSAL_CAP_RELAX`) so thrash continues. CAP still
   deprioritizes when other open arms exist.
2. **Decisive CAP counting:** fixture-only / harness-only terminals do not burn CAP.
3. **New size-matched successors:** `literal-close-structure`,
   `literal-close-component-token`, `literal-close-typed-balance`,
   `symbol-boundary-structure`, `semantic-contrast-structure`.

## Non-goals

- Recycling multi-seed-closed approaches without new recipes
- Weakening ship gates or multi-seed close rules
