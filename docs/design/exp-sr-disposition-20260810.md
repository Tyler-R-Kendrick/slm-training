# EXP-SR cross-experiment disposition (RSP-009 / SLM-490)

**Schema:** `rsp009_cross_experiment_disposition/v1` / `mechanism_disposition_report/v1` (SGS-009)

**Matrix set:** `slm490_rsp009_disposition` — EXP-SR-1..12 catalogue families

**Claim class:** fixture / scratch / blocked evidence only; no ship-gate bypass.

## Executive finding

RSP-009 audited all twelve EXP-SR catalogue families using committed sibling
evidence under `docs/design/`. OpenUI-scoped rows are separated from
symbolic-regression rows and from the dual-scope EXP-SR-12 portability
certification. **Zero** rows satisfy `adopt_primary` or `adopt_optional` under
SGS-009 fail-closed rules — champion pointers stay empty.

## Full structured evidence

- [`iter-slm490-rsp-009-disposition-20260810.md`](iter-slm490-rsp-009-disposition-20260810.md)
- [`iter-slm490-rsp-009-disposition-20260810.json`](iter-slm490-rsp-009-disposition-20260810.json)

## Disposition counts (fixture audit)

| Outcome | Count |
| --- | --- |
| `retain_diagnostic` | 7 |
| `reject` | 3 |
| `revise_and_retest` | 2 |
| `blocked` | 1 (`exp-sr-11` PySR/SRBench — external-blocked, not reject) |
| `adopt_primary` / `adopt_optional` | **0** |

Rejected or blocked families: `exp-sr-2`, `exp-sr-5`, `exp-sr-10`, `exp-sr-11`.

## Follow-up issues

- **SLM-491** — closed `external_blocked` prepared package ([`iter-slm491-exp-sr-3-external-blocked-closeout-20260810.md`](iter-slm491-exp-sr-3-external-blocked-closeout-20260810.md)); real human calibration still blocked
- **SLM-492** — closed `external_blocked` prepared package ([`iter-slm492-exp-sr-11-external-blocked-closeout-20260810.md`](iter-slm492-exp-sr-11-external-blocked-closeout-20260810.md)); PySR/Julia still blocked

## Supersessions (SLM-491 / SLM-492)

See the **Supersessions** section in [`iter-slm490-rsp-009-disposition-20260810.md`](iter-slm490-rsp-009-disposition-20260810.md) — `exp-sr-3` and `exp-sr-11` rows link to the prepared-package closeouts without changing adopt/champion pointers.

## Reproducibility

```bash
python -m scripts.run_rsp009_disposition --mode fixture
```

This document is the stable alias pointer; follow-ups and downstream docs should
cite `docs/design/exp-sr-disposition-20260810.{md,json}` and link to the full
iter-slm490 evidence pair above.
