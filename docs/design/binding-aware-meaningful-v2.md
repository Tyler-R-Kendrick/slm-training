# Binding-aware meaningful-program metric v2

**Status:** implemented as an additive, uncalibrated diagnostic. The active
ship-gate primary remains `meaningful_program_v1`.

## Contract

`meaningful_program_v1` is the historical `_is_meaningful_program` implementation
in `eval_runner.py`. Its behavior, wire field (`meaningful_program_rate`), and
existing thresholds are preserved. It validates, rejects empty roots/cards,
requires a non-`Stack` component and a placeholder, then applies the historical
gold component-type recall floor.

`binding_aware_meaningful_v2` evaluates the raw prediction before canonical
serialization can prune evidence. It returns a typed
`SemanticMeaningReportV2` with:

- one boolean verdict;
- `PASS`, `FAIL`, `NOT_APPLICABLE`, or `UNKNOWN` for every check;
- stable reason codes and source spans or AST paths;
- prompt-contract provenance;
- component, binding, and placeholder inventories;
- metric name, semantic version, and implementation SHA-256.

The verdict is positive only when every check is `PASS` or
`NOT_APPLICABLE`. `UNKNOWN` is never positive.

| Check | Evidence and failure behavior |
| --- | --- |
| Official parse | Official `@openuidev/lang-core` parse/schema validation |
| Canonical roundtrip | `serialize(validate(x))` is idempotent |
| Prompt-relevant semantic content | Nontrivial content plus deterministic prompt component facts |
| Required inventory | Effective model-visible slot contract plus deterministic prompt component requests |
| Binding correctness | Raw definitions/references, duplicate names, unresolved names, and unreachable bindings |
| Schema/value roles | Existing official-schema semantic judge, including dynamic expression nodes |
| Whole-program verifier | Applicable G0/G1/G8 lexical, grammar, and canonical gates |
| Anti-gaming | Repeated subtrees, repeated placeholder spam, low-diversity filler, and mechanical text-only inventory coverage |

Natural-language inventory that cannot be extracted deterministically remains
`UNKNOWN`. The evaluator does not substitute hidden `record.placeholders` for a
prompt-visible contract and does not call an LLM judge.

## Aggregate fields

Every suite now reports both metric families:

- `meaningful_program_rate` and `meaningful_program_v1_rate` — identical v1
  values for compatibility;
- `binding_aware_meaningful_v2_rate_strict` — positives divided by all document
  rows, so unknown rows remain negative;
- `binding_aware_meaningful_v2_rate_coverage_conditioned` — positives divided by
  rows with no `UNKNOWN` check;
- `binding_aware_meaningful_v2_coverage` — covered rows divided by all document
  rows;
- `meaningful_metric_primary=meaningful_program_v1` and version/hash metadata.

Quality/grammar summaries and dashboard/report schemas carry these fields
additively. Grammar halving and ship gates continue to use v1. The v2 policy
entry has no threshold; a follow-on calibration must select and version a
threshold instead of copying the v1 bar.

## Interpreting compact alternatives

Grammar validity, prompt-contract validity, and resemblance to the single gold
program are different claims:

- `placeholder_fidelity` measures coverage of authoritative slots. It is not a
  whole-program similarity score.
- `structural_similarity` and `component_type_recall` are explicitly
  gold-reference-relative. A smaller but prompt-equivalent implementation can
  score below 1.0 on either metric.
- v1 also embeds a 0.50 gold component-type recall floor, so a v2-valid compact
  alternative can theoretically fail the active v1 meaningful gate.
- v2 uses the prompt-visible component and slot contract and preserves minimal,
  alpha-renamed, and canonical-order variants in its gaming corpus. It remains
  diagnostic until the blinded EFS0-04 calibration supplies independent labels.

Reference-relative metrics remain useful density and regression diagnostics,
but they are not proof that an otherwise strict-valid program is semantically
wrong. Dashboard labels call this distinction out; ship thresholds remain
unchanged until calibration.

## Deterministic gaming suite

The committed
[`meaningful_v2_gaming.jsonl`](../../src/slm_training/resources/evals/meaningful_v2_gaming.jsonl)
covers empty valid roots, irrelevant populated programs, wrong placeholder
identity and property roles, duplicate/filler spam, schema-role errors, wrong
references, dead content, prompt negation, minimal positives, alpha-renamed
positives, and canonical-order variants. Each negative pins at least one
expected v2 reason code.

## Frontier replay

Use the canonical audit CLI to rescore stored generation sets rather than
regenerating them:

```bash
python -m scripts.audit_meaningful_program \
  --replay-bundle src/slm_training/resources/evals/meaningful_v2_frontier_replay.json \
  --minimum-frontier-sets 10 \
  --output docs/design/iter-efs0-03-meaningful-v2-frontier-audit-20260717.json \
  --run-dir outputs/runs/efs0-03-meaningful-v2-frontier
```

The committed bundle contains raw predictions, source records, reconstructed
effective requests, content digests, and capture-time checkpoint hash
verification for 11 sets. Replay therefore does not depend on gitignored
`outputs/`. Stored predictions at the legacy 500-character detail cap without
a digest are truncated/unknown. Human, AgentV, or EFS0-04 labels are compared
only when present with provenance; missing labels are reported, never inferred.

## Scope and limitations

V2 is a deterministic necessary-condition metric, not a theorem of intent
equivalence. Direct schema component mentions and prompt-visible slot contracts
are the hard semantic contract; open-ended intent remains unknown. The gaming
detectors are deliberately narrow and evidence-bearing. Fixture results prove
wiring and attack detection only, not model readiness. Production claims still
require the complete multi-suite scoreboard and unchanged `--ship-gates`.

## Measured results — deterministic gaming audit (2026-07-17)

The committed [result JSON](iter-efs0-03-meaningful-v2-gaming-20260717.json)
records the exact replay. Recipe: CPU, deterministic evaluator (no model or
checkpoint), `meaningful_v2_gaming.jsonl`, `n=16`, all prompt contracts hard,
and the metric implementation SHA-256 recorded in JSON. Command:

```bash
python -m scripts.audit_meaningful_program \
  --gaming-corpus src/slm_training/resources/evals/meaningful_v2_gaming.jsonl \
  --minimum-frontier-sets 0 \
  --output docs/design/iter-efs0-03-meaningful-v2-gaming-20260717.json \
  --run-dir outputs/runs/slm105-meaningful-v2-gaming
```

| Evidence | Result |
| --- | --- |
| Expected corpus outcomes | **16/16 matched**; zero replay failures |
| V2 strict / covered rate | 0.3125 / 0.3125 (five intentional positive controls) |
| Coverage | 1.00 |
| AgentV | **1/1 JSON envelope validated**, zero execution errors; semantic verdict comes from replay |
| Ship verdict | **Not evaluated** — deterministic fixture evidence is wiring and anti-gaming coverage only |

The separate
[frontier audit](iter-efs0-03-meaningful-v2-frontier-audit-20260717.md)
replays 11 distinct checkpoint generation sets (`n=33`) from the committed
bundle. Seven historical v1 positives do not clear v2. E230, E240–E247,
E288-fixed, E292, and X22 remain explicit `UNKNOWN`; independent judge, human,
per-case AgentV semantic, and EFS0-04 labels are also `UNKNOWN` with `n=0`.
