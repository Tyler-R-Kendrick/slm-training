# DSH3-12 collapsed operator traces (SLM-380)

Date: 2026-07-23
Status: implemented; strict fixture contract passed
Scope: CAP2 symbolic trace composition; no model, checkpoint, or ship claim

## Decision

Verified multi-turn AST-edit traces may now become single symbolic
multi-instruction examples only when ordinary conversation replay proves the
source trace first. `CollapsedInstructionV1` retains every source turn,
operator application, root/final state identity, state/AST digest, and the
required operation order.

The corpus schema is closed. Its question contains only
`APPLY_OPERATOR_SEQUENCE`, the root AST, and an explicit integer order. Its
answer contains only the serialized operator sequence and the trace-authoritative
final AST. SLM-379 and CERT_CAP1 are unavailable, so this change emits no
natural-language instruction or target.

## Admission and refusal boundary

Collapse first replays the entire immutable trace through state-specific pack
authority. It then rejects:

- traces shorter than two AST edits;
- undo, redo, fork, or checkout history operations;
- no-op transitions;
- cycles back to an earlier canonical state;
- repeated application identities; and
- any final-state disagreement with authoritative replay.

For each adjacent pair that lacks a mutual commuting declaration, the
transform actually applies the reordered sequence. A rejected application is
persisted as a typed conflict. A replayable reorder is a hard negative only
when its final canonical state differs. An equivalent reorder is retained as
equivalent and never mislabeled as negative. Required order is therefore
measured, not inferred from declaration names.

## Strict fixture run

The final evidence run used CPU, the strict profile, fixture source, no
synthesizer, two roots, two actions per state, and a 32-combination
per-operator bound. It completed in 5.4 seconds, inside the three-minute cap.

| Measure | Result |
| --- | ---: |
| Train candidates / admitted / rejected | 20 / 19 / 1 |
| Source operator records / roots | 20 / 2 |
| Collapsed multi-instruction records | 2 |
| Source turns per collapse | 2 |
| Exact final-state matches | 2 / 2 |
| Explicit-order records | 2 / 2 |
| Reordered hard negatives | 2 |
| Reordered conflicts (`ref.missing`) | 2 |
| Natural-language records | 0 |
| Invalid generated families | 0 |
| Mean admitted source quality | 1.0 |

The collapsed-corpus fingerprint is
`3d4663d55e5a1fa54a86aabc391c8116b7a8c3e0c8a0cd7123f0a51003561cc4`.
The parent strict-build fingerprint is
`e086b62faf8cecb326a5697ecb12e5f7e6af5bc2e34e922dc3be1cafb9510928`.
The complete measured record is
[`dsh3-12-collapsed-operator-traces-20260723.json`](dsh3-12-collapsed-operator-traces-20260723.json).

This is fixture/wiring evidence. It proves deterministic schema construction,
exact sequential replay, explicit order retention, and reordered-negative
classification on the configured roots. It does not establish model planning
quality, broad operator coverage, or a ship gate.

## Synthesis feedback

The parent build rejected only `train_text_only_01` at
`decontamination/test_fixture_structure`. `quality_report.json` contained no
warnings. `synthesis_feedback.json` repeated the existing
`eval_leakage_source` hypothesis for the `human_curated` fixture family.
The reproduction was attached to SLM-392; no duplicate issue was created and
the strict firewall remains unchanged.

## Validation

Focused controls cover exact two-turn collapse, equivalent reorder
preservation, short-trace refusal, history-operation refusal, cycle refusal,
closed corpus keys, explicit order, trace-authoritative final AST, pipeline
artifact registration, deterministic fingerprints, and version stamps.

No checkpoint was created, so the model card and README checkpoint summary do
not change. This was a data build rather than an evaluation, so AgentV output
is not applicable.

## Research lineage

[Saha and Kanewala, 2018](https://arxiv.org/abs/1802.07361) motivates
systematic source/follow-up test construction for metamorphic fault detection.
DSH3-12 is an adapted repository contract: the paper does not define OpenUI
operators, immutable conversation traces, canonical AST equality, explicit
instruction order, or this hard-negative schema, and no paper result is
reproduced here.
