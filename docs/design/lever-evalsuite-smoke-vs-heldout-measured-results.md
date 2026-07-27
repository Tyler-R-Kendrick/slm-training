# Eval-suite-size lever: smoke n=3 vs held_out n=5 — NOT SHIP

**Honesty:** `fixture_or_scratch` / diagnostic smoke + held_out at tiny n (3/5).
**Not a ship claim** (single-suite runs, no `--ship-gates`, not the full
5-suite scoreboard).

## Hypothesis

At the frozen champion micro-recipe (s16 · lr=1e-3 · bs=2 · structural_bias=1.5
· seed=47 · ASAP · t30 · scratch/cpu), the smoke suite's
`meaningful_program_rate=0.333` (n=3) reflects real headroom that persists or
improves on a larger held-out suite, rather than n=3 sampling noise. This is
the "measure meaningful headroom on a larger suite than n=3" next-lever named
in
[`lever-train-corpus-e1291-e937-vs-smoke-measured-results.md`](lever-train-corpus-e1291-e937-vs-smoke-measured-results.md).

## Recipe

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 0.001 --structural-bias 1.5 --seed 47 --device cpu --asap-decode \
  --run-id exp_lever_evalsuite_s16_lr1e3_bs2_sb15_seed47 --no-sync-checkpoints

slm data build-test --source fixture --no-rico-path --version wf_smoke_v2 \
  --train-manifest src/slm_training/resources/data/train/wf_smoke_v2/manifest.json

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/wf_smoke_v2 --suite {smoke,held_out} \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_evalsuite_s16_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix --eval-limit {3,5}
```

`slm data build-test` produced `suite_counts`: smoke=3, held_out=5,
adversarial=4, ood=4, rico_held=0.

## Train outcome

| run_id | train_dir | records | stopped_on | last_loss | wall_s |
| --- | --- | ---: | --- | ---: | ---: |
| `exp_lever_evalsuite_s16_lr1e3_bs2_sb15_seed47` | `src/slm_training/resources/data/train/wf_smoke_v2` | 101 | steps | 12.85036563873291 | 3.21 |

## Results (same checkpoint, two suites)

| suite | n | parse | exact_match | meaningful | reward | empty | timeout | placeholder_fidelity | structural_similarity | lat_p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.3333333333333333 | 0.7833333333333333 | 0 | 0 | 0.5277777777777778 | 0.25416666666666665 | 30002.79 |
| held_out | 5 | 1.0 | 0.0 | 0.0 | 0.694 | 0 | 0 | 0.29 | 0.09758 | 30002.1 |

### Confound

`held_out` differs from `smoke` in **both** suite domain and `n` (5 vs 3) —
this is a suite-generalization check, not a pure n ablation. This fixture
build has no same-domain suite at n>3 to isolate the n effect alone.

## Decision

**REJECT** the headroom hypothesis — `meaningful_program_rate` **collapses**
0.333 (smoke n=3) → **0.0** (held_out n=5) on the identical checkpoint.
`placeholder_fidelity` (0.528→0.29) and `structural_similarity`
(0.254→0.098) both drop substantially too, while `parse_rate` holds at 1.0 on
both suites.

The smoke n=3 meaningful-program result **does not generalize** to a
different, still-tiny held-out suite. This reinforces (does not contradict)
`honest-ship-eval`'s fixture-demo-vs-ship distinction, and must **not** be
read as new headroom to chase via more micro-recipe tuning.

## Reproducibility finding (side finding, not the lever under test)

The currently-committed `src/slm_training/resources/data/train/wf_smoke_v2`
fixture (**101** records) is **not** the corpus behind the historical
`wf_smoke_v2` champion baseline cited in prior lever docs — see
[`lever-train-corpus-e1291-e937-vs-smoke-measured-results.json`](lever-train-corpus-e1291-e937-vs-smoke-measured-results.json)
(`arm: wf_smoke_v2`, `record_count: 103`, `last_loss: 7.451866626739502`).

`outputs/` is gitignored/ephemeral per session, so that original 103-record
build is not reproducible from source control. This session's fresh
train-on-committed-fixture run instead reproduced `last_loss=12.85036...`,
matching the separately-documented **`lever_fixture_v1`** fresh-fixture-rebuild
lever
([`lever-fresh-fixture-build-vs-smoke-measured-results.md`](lever-fresh-fixture-build-vs-smoke-measured-results.md)),
not the original champion snapshot.

**Action needed (not fixed in this pass):** resolve the `wf_smoke_v2`
naming/reproducibility drift — the committed resource no longer matches what
prior docs call `wf_smoke_v2` — so future lever docs cite a corpus that is
actually reproducible from this checkout.

## Next lever (evidence-backed)

1. Resolve the `wf_smoke_v2` fixture drift above before running further
   comparisons against the old champion baseline numbers.
2. Pursue real data-quality work via the `synthesis-feedback` loop
   (`abstraction_ladder` exposure yield=0, noted in the fresh-fixture-build
   doc) rather than further n≤5-scale hyperparameter sweeps on fixture
   suites.

Captured: 2026-07-27T16:47:06.689Z
