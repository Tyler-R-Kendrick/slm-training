# Autotrain c1771: bounds fixture efficiency candidate

**Verdict:** queue only for fresh confirmation. Grammar completion bounds held
all measured smoke quality exactly while improving fixture meaningful-programs
per millisecond by 6.16%, just above the 5% screening threshold. This is a
three-record local screen, not a confirmed performance or ship result.

| Arm | Params / train | Smoke quality | Runtime | Decision |
| --- | --- | --- | --- | --- |
| bounds | 1,608,962; 21 steps; loss 20.22257; 3.18 s | parse 1; meaning .3333; structure .11527; binder F1 .82222; fidelity .72222; reward .82367 | p50 1,433.96 ms; 136 forwards; 35,792 unique states | queue confirmation |
| matched control | 1,608,962; 21 steps; loss 20.22257; 3.02 s | exact metric and prediction tie | p50 1,522.25 ms; identical forward/state work | baseline |

AgentV completed both bundles with zero execution errors. Honest gates still
fail: smoke n=3 is below the required volume, meaning/structure/recall remain
below thresholds, and held-out/adversarial/OOD/rico_held suites were not run.
The primary structural metric did not improve.

This cycle also revealed an orchestration gap: c1770's observed, confidence
.95 literal-close successor was discarded by generic recent-arm cooldown, so
the loop ran bounds instead. Campaign orchestration v77 gives an explicit,
high-confidence observed `next_experiment` priority precedence over cooldown;
ordinary speculative or rejected arms remain cooled down.

Both checkpoints are local CPU scratch artifacts with explicit no-sync policy.
They are not reusable, promoted, or ship candidates. Lean is
`not_applicable:screening`; if fresh confirmation succeeds, the existing
promotion path must run formal preflight before promotion.

Machine evidence:
[`autotrain-cycle-1771-bounds-efficiency-candidate.json`](autotrain-cycle-1771-bounds-efficiency-candidate.json).
