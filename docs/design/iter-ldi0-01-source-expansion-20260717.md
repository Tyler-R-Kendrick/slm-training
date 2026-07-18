# Iter LDI0-01 — local-decision architecture contract + source expansion (2026-07-17)

**Issue:** SLM-114 (LDI0-01). **Type:** documentation / research-inventory / test
contract. **No code, model, checkpoint, adapter, or model-card quality change.**

**Reviewed base commit:** `c7cb099` (`origin/main`, "Cache exact choice completion
states (#315)").

## What changed

Made the local-decision-intervention (LDI) research synthesis and architectural
boundaries canonical in the repository so future agents cannot recreate a parallel
harness, repeat the falsified E249-E284 chain, or treat a local-metric gain as
promotion evidence.

- [`local-decision-interventions.md`](local-decision-interventions.md) — added the
  **Architecture contract (LDI0-01)** (evidence / objective / actuator / experiment /
  promotion separation with named owners; six invariants; forbids a second stack;
  `DecisionEventV2` as the next contract) and a **Measured chain and current
  blocker** section (E248-E286 negative chain; blocker = *stable state support does
  not imply objective/action-partition support*).
- [`local-decision-sources.json`](../../src/slm_training/resources/autoresearch/local-decision-sources.json)
  — added nine academic works (see table).
- [`research-lineage.md`](research-lineage.md) — extended the *Exact-state local
  decision preference* entry with the new sources under the existing
  Faithful/Adapted/Surrogate/Adjacent vocabulary; no duplicate entries.
- [`quality-experiment-matrix.md`](quality-experiment-matrix.md) — added the **LDI
  campaign index** (namespace pointer only; claims no unrun row and mints no E ID;
  records the E-ID allocation rule with next-free `≥ E292`).
- `tests/test_autoresearch/test_harness.py` — updated and extended
  `test_local_decision_source_manifest_is_complete` (counts, source-ID uniqueness,
  required metadata, unique canonical URIs, implementation-status vocabulary, the
  nine additions).

## Source count before / after

| | Rows | Academic (arXiv) | Implementation |
| --- | --- | --- | --- |
| Before | 33 | 25 | 8 |
| After | 42 | 34 | 8 |

## Duplicate / alternate-URL handling

Each addition is a distinct canonical arXiv `abs/` URI with a unique `source_id`;
none duplicates an existing paper or an alternate URL. The pre-existing alternate
URLs (DeepSeek-R1 Nature DOI, two OpenReview forum links) remain in `metadata`
rather than as separate rows. The manifest test asserts
`len({source_id}) == len({uri}) == len(rows)`.

## Classification (nine additions)

Metadata verified against arXiv on 2026-07-17. `implementation_status` follows the
research-lineage vocabulary (all additions are Adjacent or Adapted; none is
implemented by this contract).

| arXiv | Work | Category | Status | LDI relevance |
| --- | --- | --- | --- | --- |
| [1810.04650](https://arxiv.org/abs/1810.04650) | MGDA (MTL as multi-objective opt.) | objective_geometry | Adjacent | Frames the objective/action-partition blocker |
| [2001.06782](https://arxiv.org/abs/2001.06782) | Gradient Surgery / PCGrad | objective_geometry | Adjacent | Diagnostic lens for the same blocker |
| [2106.09685](https://arxiv.org/abs/2106.09685) | LoRA | adapter_actuator | Adjacent | Removable causal actuator (later), not the event schema |
| [2109.05093](https://arxiv.org/abs/2109.05093) | PICARD | constrained_decoding | Adjacent | Prior art: hard constraints stay deployed |
| [2303.10512](https://arxiv.org/abs/2303.10512) | AdaLoRA | adapter_actuator | Adjacent | Deferred adaptive-rank actuator |
| [2402.03300](https://arxiv.org/abs/2402.03300) | DeepSeekMath / GRPO | verifiable_training | Adjacent | Deferred RLVR behind the readiness gate |
| [2405.21047](https://arxiv.org/abs/2405.21047) | Grammar-Aligned Decoding / ASAp | constrained_decoding | Adjacent | Legality ≠ preference (invariant 1) |
| [2407.01082](https://arxiv.org/abs/2407.01082) | Min-p sampling | decoding_sampling | Adjacent | Decoding-time baseline to beat |
| [2603.00025](https://arxiv.org/abs/2603.00025) | TAB-PO | local_preference | Adapted | Token-critical PO informs DecisionEventV2 weighting |

## Honesty boundary

Specification / inventory only. No experiment, train, eval, benchmark, checkpoint,
adapter, or model-card quality update. No paper result is represented as a
repository result; every source carries a lineage label, and constraint shadows
certify decoder legality only — never a semantic preference label. No ship gate is
weakened and no readiness or promotion claim is made.
