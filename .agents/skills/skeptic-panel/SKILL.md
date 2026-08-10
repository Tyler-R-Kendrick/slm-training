---
name: skeptic-panel
description: Use when a screening positive appears in the continuous autotrain loop and before any confirmation spend — convene three independent skeptic lenses (statistical power, prior evidence, mechanism plausibility) to attempt refutation; majority-refute files the candidate as rejected with reasons
---

# Skeptic panel (refutation before confirmation spend)

## Overview

A screening positive is a **claim to be refuted**, not a result to be
confirmed. RC3 (`docs/design/harness-evolution-architecture-review-20260809.md`)
showed the loop rediscovering and rejecting the identical delta
0.3267→0.3828 in nine separate loops because nothing tried to kill the claim
before spending confirmation cycles. This skill runs that kill attempt.

**When:** any screening positive (`sdlc_delivery.json` with `positive: true`
and `cycle_intent: screening`), BEFORE the champion queue spends a confirm or
promote cycle on it.

## Inputs

1. `python scripts/query_evidence.py` output for the candidate's
   `config_fingerprint`, lever keys, and endpoint metric (durable prior
   attempts across sessions).
2. The cycle's preflight verdicts — the `preflight` field of the campaign's
   `sdlc_delivery.json` / `outputs/autoresearch/sdlc_delivery_ledger.jsonl`
   entry (verdicts from `slm_training.autoresearch.preflight`, see
   `docs/design/preflight-gates.md`).

## Panel protocol

Convene **three independent skeptics**. Each writes its verdict
(`refuted` / `not_refuted`) with reasons BEFORE reading the other two — no
anchoring, no shared drafting.

### Skeptic 1 — statistical power lens

- Could this measurement have rejected the null at all? (n paired documents,
  `power_feasibility_report` in `slm_training.autoresearch.evidence_ledger`.)
- Is the observed delta within the noise band of prior same-arm deltas in the
  evidence ledger? Single-seed fixture n is noisy by construction.
- Was the measurement complete (`measurement_complete: true`, no decode
  timeouts diluting the metric)?

### Skeptic 2 — prior evidence lens

- Has this `config_fingerprint` + endpoint been attempted before? With what
  outcome and power? An identical config already `confirm_failed` /
  `ship_rejected` with adequate power is a refutation by record.
- Is this delta suspiciously identical to a previously rejected one (the RC3
  signature)? Same magnitude + same endpoint + same lever family ⇒ presumed
  rediscovery until shown otherwise.
- Do `warn` preflight verdicts indicate underpowered prior attempts this run
  did not add power to?

### Skeptic 3 — mechanism plausibility lens

- Is there a causal story from the lever moved to the endpoint metric that
  respects the repo's architecture invariants (constrained decoding is the
  product; capability is never bought with parameters)?
- Could the delta be an artifact: control mismatch, eval-partition drift
  (`version_stamp` eval components differ), steps/seed jitter, or a latency
  win masquerading as quality?
- Does the win survive the non-regression metrics (parse_rate,
  binder_reference_f1)?

## Decision rule

- **Majority refute (≥ 2 of 3)** ⇒ file the candidate as rejected: record the
  panel's reasons in the delivery record (append to `reasons`, keep
  `positive` reclassified honestly), do not enqueue/confirm the champion, and
  ensure the outcome lands in the evidence store so the preflight
  `prior_attempts` check blocks the rediscovery next time.
- **Otherwise** ⇒ the candidate proceeds to the normal confirmation path with
  the panel's doubts attached as reasons.

## Forbidden

- Weakening any gate, bar, alpha, or minimum_effect to let a candidate pass.
- Softening a skeptic's lens because the delta "looks exciting".
- Skipping the panel because preflight already passed — preflight gates spend,
  the panel gates *belief*.
- Re-running the measurement until it confirms (that is p-hacking, not
  skepticism).

A rejected candidate closes an approach, never a goal.
