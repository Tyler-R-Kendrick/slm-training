# Harness artifact audit (SLM-281)

SLM-281 provides a byte-preserving replay protocol for archived evaluation
failures. It preserves the stored prediction bytes locally, records their
SHA-256, and never regenerates a model output. Archive-derived reports must
remain under local `outputs/` unless the source data is explicitly approved for
publication.

Each replay record carries `HarnessProvenanceV1`: source-evaluation digest,
evaluation policy, timeout, canvas cap, parser fallback, repair policy,
target length, browser, runtime, verifier, and raw/constrained/repaired
identifiers. Canonical model-build evaluation artifacts emit one suite-level
provenance record plus a stable provenance ID on every detail row. Older
records without those fields retain `unknown_not_captured`; replay-time
feasibility is not attributed to the original decoder.

The harness can classify byte-preserving failures as stable, timeout-, canvas-,
or truncation-sensitive. It records `actual_decode_replayed=false`: archived
outputs lack the original decoder trace, so a feasibility sweep is not a
causal re-decode experiment.

## Local byte re-verification — 2026-07-24

[`iter-slm281-harness-replay-20260724.json`](iter-slm281-harness-replay-20260724.json)
is an authorized, redacted local re-verification result. It contains no raw
prediction bytes or serialized programs: rows retain only stable IDs, hashes,
provenance IDs, and aggregate verifier facts.

The capped CPU run selected 100 archived failures — 20 each from `smoke`,
`held_out`, `adversarial`, `ood`, and `rico_held` — by deterministic
source/failure/length/decode-path strata. It reused production oracle scoring,
the OpenUI verifier cascade, and the local Chromium preview verifier, with
target/canvas feasibility arms at 64, 128, and 256 tokens. AgentEvals and the
pinned AgentV SDK passed both automatic evidence checks (2/2); no human rating
was requested or used.

| Measure | Result |
| --- | --- |
| Raw bytes preserved / model output regenerated | 100/100 / no |
| Byte-reverification label flips | 0/100 (0.0%) |
| Complete linked decoder traces / required | 0/100 |
| Partial legacy prefix links (diagnostic only) | 8/100 |
| Architecture claims | automatically blocked |

The 64-token canvas cannot satisfy the `held_out`, `ood`, or `rico_held`
meaningful-program feasibility floors; 128 still cannot satisfy `rico_held`;
256 clears the feasibility check. These are output-independent feasibility
facts, not evidence of what a decoder would have produced.

The result is not a model or ship claim. Archived outputs do not retain
complete, record-linked decoder decisions, so decoder timeout, fallback, and
repair flip rates remain `null`, and the automatic block remains in force even
though the byte re-verification flip rate is below 15%. This is an evidence
rule, not a human gate.
