# hillclimb_strict_v1 train snapshot (SWARM-DATA)

Machine-readable: [hillclimb-strict-v1-train-build.json](hillclimb-strict-v1-train-build.json).

**Honesty:** fixture-scale strict snapshot for autotrain hill-climb recovery. Not a production HF ship corpus. `wf_smoke_v2` (101 records) is unchanged for replay.

## Integrator policy one-liner (SWARM-METRIC)

`defaults.train_version: "hillclimb_strict_v1"`

Do not edit `policy.v2.json` in this swarm; apply that one field bump in the Integrator / SWARM-METRIC patch.

## Recipe

```bash
python -m scripts.build_train_data \
  --profile strict --source all --version hillclimb_strict_v1 --immutable \
  --synthesizer quality --programspec-count 48 --max-children 6 \
  --ngram-decontam --decontam-eval-root src/slm_training/resources/data/eval \
  --publish --register-lineage
```

| Field | Value |
| --- | --- |
| Device | CPU |
| `MAX_RUN_MINUTES` | 3 (build finished inside cap) |
| Profile | strict (fuzzy + semantic dedup, Bronze floor, n-gram decontam) |
| Source | `all` |
| Synthesizer | `quality` (layout aug + templates) |
| Published | `src/slm_training/resources/data/train/hillclimb_strict_v1/` |
| `record_count` | **976** (target ≥1000; max admitted under strict gates in this run) |
| Mean quality | 0.9756 |
| Parse rate | 0.9669 |
| Judge pass | 1.0 |
| Candidates / rejected | 2478 / 1513 |
| N-gram decontam drops | 102 (gate held) |
| Preference pairs | 308 |

`--dedup-against` accepts **train** dataset ids only; eval disjointness is the strict n-gram decontam walk over `resources/data/eval` (includes `e938_role_safe_all_targets_smoke6_v1` and `e938_role_safe_all_targets_v2`).

## Leakage (fail closed)

Exact prompt⊕openui pair overlap vs both e938 eval suites: **0**. Id overlap: **0**. Verdict: **PASS**.

## Synthesis loop

Top named synthesizer: **`template`** (579 candidates, yield 0.2712, dominant reject `exact_pair_duplicate`). Exclusive SWARM-DATA ownership forbids editing `harnesses/train_data/synth.py`; the follow-up is reduce template expansion / diversify namespaces (skill `synthesis-feedback` `redundant_expansion`). This snapshot keeps the 157 admitted template rows.

Also filed: `awwwards*` yield 0 (quarantine G10), `lexical_typed_map` yield 0 (`parse_or_contract_error`). Gates not relaxed.

26 `experiment_candidates` from `synthesis_feedback.json` copied into
`outputs/autoresearch/loops/continuous-openui-local/synthesis_experiment_candidates.jsonl`
(gitignored loop evidence) and into the JSON sibling of this note.

## Tests / validation

- Build's own strict gates (quality, fuzzy/semantic dedup, n-gram decontam).
- One-command exact-pair leakage check vs e938 smoke6 + v2: PASS.
- `DataStore.versions("train")` resolves `hillclimb_strict_v1` (`record_count` 976).
- No scratch optimizer train in this swarm (would compete with SWARM-CHAMPION for the 3-minute cap); consumption is the published snapshot + store resolve.

`version_stamp` in the JSON sibling (`harness.train_data` v32). No `versions.json` bump in this swarm.
