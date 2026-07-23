# SLM-392 human-curated fixture source overlap audit

Date: 2026-07-23
Status: source remediated; strict rerun passed
Scope: fixture-source decontamination; no model, checkpoint, or ship claim

## Finding

`train_text_only_01` was an accidental train-source duplicate of reserved test
structure. Its raw source used `Stack([blurb], "column")`, while the committed
adversarial fixture `adv_empty_prompt_01` used the implicit Stack default.
Strict sanitization correctly elided the explicit `"column"` argument. Both
then normalized to:

```text
id = Stack([id]) id = TextContent(":ph")
```

Both sanitized structures have fingerprint
`198ad03f517e14bd83dc92db0c458058d2f9b5671db1f1ac95e5544e027801a8`.
The firewall rejection was therefore correct. The source seed, rather than the
gate or threshold, was wrong.

## Intervention

Remove `train_text_only_01` from the canonical train seed source. Retain
`adv_empty_prompt_01`, the reserved-structure firewall, strict sanitization,
8-gram decontamination, the 0.5 overlap threshold, and every other admission
policy unchanged.

A regression test now compares every committed train fixture, after strict
sanitization, with both raw and strictly sanitized reserved test structures.
This catches source contamination before the downstream build has to reject it.
Historical immutable corpora are not rewritten.

## Matched strict rerun

The rerun reused the DSH3-10 CPU fixture recipe: strict profile, fixture source,
no synthesizer, two operator roots, two actions per state, 32 bounded
combinations per operator, sibling forks, no publish, and no lineage
registration. It completed inside the three-minute cap.

| Measure | Before | After |
| --- | ---: | ---: |
| Source candidates | 20 | 19 |
| Admitted records | 19 | 19 |
| Rejected records | 1 | 0 |
| Reserved-structure drops | 1 | 0 |
| N-gram decontamination drops | 0 | 0 |
| Mean admitted quality | 1.0 | 1.0 |

The admitted content fingerprint remains
`e086b62faf8cecb326a5697ecb12e5f7e6af5bc2e34e922dc3be1cafb9510928`.
Removing the already-rejected source row therefore changes no admitted training
content.

The sibling operator build still emits 20 records from two roots, including two
collapsed records, with 27 legal successes, 533 retained rejected combinations,
zero illegal targets, and zero invalid families. This is fixture wiring
evidence only.

## Quality-loop disposition

The required post-build inspection found:

- `quality_report.json`: zero warnings, reserved-structure drops, and n-gram
  drops;
- `rejected.jsonl`: empty;
- `synthesis_feedback.json`: `human_curated` yield 19/19 (1.0), zero warnings,
  zero recommendations, and zero new experiment candidates.

The hypothesis passes. The source contamination is removed without weakening a
gate, changing admitted content, creating a checkpoint, or making a model or
ship claim. Machine-readable evidence:
[`slm392-human-curated-source-overlap-20260723.json`](slm392-human-curated-source-overlap-20260723.json).
