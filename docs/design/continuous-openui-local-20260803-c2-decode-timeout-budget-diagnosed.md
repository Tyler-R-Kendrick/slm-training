# Autotrain continuous-openui-local c2: symmetric decode timeout, budget-diagnosed (not a repair)

**Status:** blocked on `repair_harness`; documenting a bounded diagnostic, not
landing a fix. Do not keep re-attempting the same policy change automatically.

## What happened

Cycle 2 (`continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`, seed
`100002`) screened `component-plan` (1,755,760 params) against its matched
control at the same size. **Both** arms recorded `decode_timeout_count=3` on
every smoke record (`completed_document_n=0/3` each), under the governed
screening policy's `screening_decode_timeout_seconds=24`
(`src/slm_training/resources/experiments/autotrain_climb/policy.v1.json`,
`measurement.screening_decode_timeout_seconds`). AgentV finalized every
record's disposition (no execution errors), so this is a typed timeout, not a
crash — but with both control and candidate timing out identically, there is
no model-attributable primary-metric comparison this cycle
(`primary_metric_unavailable`).

This reproduces the general class of decode-timeout pathology already on
record from cycle-c5 (seed `100005`,
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)),
now at a different seed (`100002`) and in a fresh local sandbox this session
bootstrapped from scratch (torch/AgentV reinstalled; no prior `outputs/`
state carried over). The prior doc explicitly asked for dedicated profiling
before any further routing change; the diagnostic below is exactly that,
scoped to evidence only.

## Diagnostic (bounded, no code change)

Re-ran `evaluate_model` directly against the cycle-c2 control checkpoint,
same seed `100002`, `--evaluation-policy strict_compiler_tree`, holding every
other recipe field fixed and raising only `--decode-timeout-seconds` from the
governed `24` to `90`:

```
python -m scripts.evaluate_model --checkpoint <c2-control checkpoint> \
  --suite smoke --evaluation-policy strict_compiler_tree \
  --seed 100002 --decode-timeout-seconds 90 --device cpu
```

Result: `completed_document_n=3/3`, `incomplete_document_n=0` — the same
records that produced 3/3 typed timeouts at the governed 24s budget complete
cleanly with more wall room. This is budget-limited decode (consistent with
the SLM-303 census/budget-sweep methodology,
[`scripts/run_slm303_decode_budget_audit.py`](../../scripts/run_slm303_decode_budget_audit.py)),
not an infinite loop, execution error, or grammar-constraint deadlock.

## Why this is not landed as a fix here

`screening_decode_timeout_seconds` (and `promotion_decode_timeout_seconds`)
are governed policy thresholds shared by every continuous cycle, historical
and future — not a per-cycle recipe knob. Per the continuous-mode contract
("Automated promotion and judge changes"), a threshold change needs its own
preregistered meta-campaign against unchanged held-out controls, not a
same-cycle edit motivated by one seed's diagnostic. Unilaterally raising it
here would silently recalibrate every historical screening/promotion
decision's timeout-vs-quality tradeoff without that campaign. This session's
remaining scope does not cover authoring that meta-campaign.

## Recommendation

File a dedicated preregistered meta-campaign (owner: `improve-openui-harnesses`
+ `improve-lean-optimums` if a Lean band is bound to the current threshold) to
evaluate raising `screening_decode_timeout_seconds` from 24 toward the 60-90s
range demonstrated here, sized against the `screening_stage_wall_minutes: 3`
cap (`MAX_RUN_MINUTES`) with both arms' train+eval still fitting inside it,
and with held-out controls proving no quality-relevant behavior is being
laundered through a longer clock. Until that lands, treat symmetric decode
timeouts at fixture scale as `harness_failure` / `measurement_incomplete`
evidence per cycle, not as a model result, and do not resurrect a routing-only
auto-retire (already tried and reverted per the cycle-c5/c6 record).
