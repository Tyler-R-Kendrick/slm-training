# SLM-286 ship-gate evidence census

- Source revision: `8c20bce30842588c5b2ca9b410f68606dfc162e8`
- Census implementation revision: `8c20bce30842588c5b2ca9b410f68606dfc162e8`
- Committed JSON artifacts scanned: `780`
- Scoreboards replayed: `439`
- Suite rows replayed: `904`
- Suite rows below `DEFAULT_MIN_SUITE_N` before reading quality: `869`
- Scoreboards with every present suite below the minimum: `408`
- `supported_negative`: `0`
- `inconclusive_until_powered`: `101`
- `invalid/confounded`: `338`
- Claimed positives/negatives with assessable paired intervals: `0`
- Claimed positives/negatives with overlapping intervals: `0`

## Recipe and decision

- Device/backend: CPU-only Git object replay; no model or checkpoint.
- Matrix set: committed `docs/design/*.json` explicit boards plus the canonical historical normalizer.
- Honesty mode: current ship policy replay; evidence reachability before quality.
- Result: historical negative labels are not currently supportable as model-quality negatives.
- Ship/default/promotion: none.

## Provenance and limitations

- Every adjudication row binds the source commit, file SHA-256, JSON pointer, canonical scoreboard SHA-256, and gate-replay SHA-256.
- Source artifacts are immutable; verified prior ledger bytes remain a prefix and corrections append superseding events.
- Unsupported artifacts are counted as exclusions; canonical nested and legacy single-suite records are normalized once.
- Overlap is computed only for declared control/candidate-style pairs whose selected rate evidence carries exact binomial counts and interval bounds. It is descriptive, never a significance test.
- OpenWiki source instructions were updated; local regeneration was not claimed because the non-interactive provider token was unavailable.

Interval overlap is descriptive only and is never treated as a significance test.
Historical source artifacts were not modified; adjudications are append-only rows in the JSON census.

## Statistical lineage

- Wilson (1927), DOI: https://doi.org/10.1080/01621459.1927.10502953
- Hoenig and Heisey (2001), DOI: https://doi.org/10.1198/000313001300339897
