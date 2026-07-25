# Compiler-inverted ProgramSpec corpus (SLM-267 / VSD2-02)

Status: **wiring-first increment; uniform-policy CLI landed; VSD-H7a
inconclusive at smoke scale with a real throughput blocker identified.**

This is the first PR against
[SLM-267 (VSD2-02)](https://linear.app/quickdeploy-ai/issue/SLM-267/vsd2-02-scale-compiler-inverted-programspec-data-to-10k100k1m-with):
scale the repository's typed `ProgramSpec`-first generator
(`src/slm_training/data/progspec/generate.py`) into a streaming,
coverage-audited semantic data engine and publish nested 10k/100k/1M corpora
under uniform and coverage-targeted sampling. That full scope is far larger
than one increment; this PR lands the streaming/resumable **uniform-valid**
control CLI the issue specifies and uses it to take an honest, measured first
reading of **VSD-H7a** ("the bounded OpenUI ProgramSpec space can supply
nested 10k/100k/1M corpora"). Coverage-targeted sampling (needs a
`CoverageGapManifestV1` consumer wired to SLM-265's outcome-blind domain-shift
strata), the 100k/1M rungs, and the in-issue training experiment are explicit
follow-on scope — not attempted here.

## What shipped

`python -m scripts.generate_programspec_corpus` (new;
`tests/test_scripts/test_generate_programspec_corpus.py`):

- Wraps the existing `ProgramGenerator` — never replaces it.
- `--policy uniform` only in this increment (`--policy targeted` is not a
  valid choice yet; the issue's `--policy` surface is preserved so a follow-up
  can add it without a CLI break).
- Deterministic `(seed, shard_id)` generation over a `GeneratorConfig`
  (`--max-depth`, `--max-width`, `--components`).
- Disk-backed **exact** dedup: the accepted set is keyed on
  `ProgramSpec.id` (already a canonical content hash of the program +
  viewport/depth/width/prop-target identity computed inside the generator) —
  no Bloom-filter-only path exists.
- Immutable rejection ledger (`rejected.jsonl`): every `generate_one()`
  `ValueError` other than grid exhaustion is appended with its call index and
  verifier-failure reason, never silently dropped.
- Resumable `state.json` (schema `programspec_corpus_stream_state/v1`) and a
  content-addressed `manifest.json` (schema
  `programspec_corpus_stream_manifest/v1`, `accepted_ids_sha256` over the
  sorted accepted-id set) under the canonical `DataStore` `programspec` root
  (`outputs/data/programspec/<dataset_id>/`) — no shadow artifact tree.
- `--describe` dry-run; `--max-wall-minutes` capped at
  `slm_training.levers.MAX_RUN_MINUTES` like every other capped harness in
  this repo.

## Determinism contract (why resume is possible at all)

For a fixed `GeneratorConfig` + `seed`, `ProgramGenerator` visits candidates in
an identical order and produces an identical accept/reject sequence on every
process invocation — nothing in candidate selection or verification consumes
wall-clock time, OS randomness, or network state. `tests/test_scripts/test_generate_programspec_corpus.py::test_resume_produces_monotonic_prefix`
locks this: resuming with a larger `--target-unique-roots` always reproduces
the prior accepted-id prefix byte-for-byte before extending it.

## Measured evidence (smoke scale, local CPU, two capped shards)

Command (both shards identical except invocation time):

```bash
python -m scripts.generate_programspec_corpus \
  --pack openui --policy uniform --target-unique-roots 10000 \
  --seed 0 --shard-id 0 --max-wall-minutes 2.5 \
  --dataset-id compiler_programs_uniform_smoke_v1
```

Config: default (all 54 pinned components), `max_depth=3`, `max_width=3` —
the same defaults `ProgramGenerator` and `GeneratorConfig` ship with.

| Shard | Calls this run | New accepted | New rejected | Cumulative accepted | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 (fresh) | 334 | 334 | 0 | 334 | `wall_clock_stopped` |
| 2 (resume) | 324 | **0** | 0 | 334 | `wall_clock_stopped` |

Full evidence:
[`iter-slm267-uniform-saturation-20260725.json`](iter-slm267-uniform-saturation-20260725.json).

Zero verifier rejections at these settings (all 334 generated programs cleared
G0–G10 as Silver) — the earlier 0/50-accept reading during local development
was an environment artifact (the `@openuidev/lang-core` Node bridge under
`src/apps/openui_bridge/` was not `npm ci`-installed), not a generator defect;
confirmed by the full `tests/test_data/test_progspec_generate.py` suite
passing once the bridge was installed.

## VSD-H7a reading: `generator_state_space_saturates` risk confirmed, not yet resolved

Two independent, measured findings bear directly on the issue's falsifier
("the state space, prompt inversion, verifier, or dedup contract is
materially narrower than assumed"):

1. **The candidate grid is a small, seed-independent, near-constant function
   of `GeneratorConfig`, not of the target root count.** Measured
   `len(ProgramGenerator(config)._candidates)`:

   | `max_depth` | `max_width` | Candidates (54 components) |
   | ---: | ---: | ---: |
   | 3 | 3 | 1,781 |
   | 5 | 4 | 1,785 |
   | 8 | 6 | 1,791 |
   | 10 | 8 | 1,793 |

   Depth/width barely move the count because the grid is dominated by
   `O(components²)` pairwise-coverage candidates and
   `O(components × properties × variants)` prop-target candidates, not by
   depth/width combinatorics. Critically, **`seed` only perturbs tie-break
   jitter among equal-coverage-score candidates — it never adds new
   candidates.** Sharding by `(global_seed, shard_id, worker_id)` as the issue
   specifies therefore cannot by itself grow reachable diversity past 1,781
   unique roots for this config; only widening `GeneratorConfig` (more
   `selected_groups`/triples, literal-value pools, multi-instance repeats
   within one program) can. That widening is unscoped in this PR.

2. **The MVP resume mechanism does not deliver practical cross-shard
   throughput at measured verifier cost.** At ≈0.45–0.46s/`generate_one()`
   call (bridge-subprocess + full G0–G10 verification per candidate), a
   from-scratch shard nets ~330 new unique roots in one capped 150s window.
   Because this increment's resume strategy re-derives state by **replaying**
   the deterministic call sequence from index 0 (skipping emission, not
   verification, for already-seen calls) rather than persisting the
   generator's internal candidate cursor, the *second* capped shard spent its
   entire budget re-executing the first shard's 334 calls and landed **zero**
   net-new records. Total work to reach cumulative count `M` across many
   small shards is closer to `O(M·shards)` than `O(M)` — the opposite of
   "resumable" at this cost level.

Neither finding is a final verdict on VSD-H7a: the smoke run used the
smallest, un-widened `GeneratorConfig` and the naive resume path. Both are
concrete, falsifiable blockers a follow-up increment must clear before a 10k
uniform corpus is reachable:

- persist the generator's internal `_used` candidate cursor (or an equivalent
  index) in `state.json` so resume is `O(new work)`, not `O(total work)`;
- widen `GeneratorConfig`/candidate construction (or add a second axis of
  variation) so the candidate grid itself can exceed 10k before the resume
  question even matters.

## Disposition

**`inconclusive`** (one of the issue's own listed acceptance dispositions).
VSD-H7a is neither confirmed nor falsified: this increment did not reach the
10k rung, but it also did not exhaust the current 1,781-candidate grid (334 of
1,781, `exhausted: false`) — it hit the resume-throughput blocker first. No
training experiment, coverage-targeted policy, or 100k/1M rung was attempted.
No checkpoint, ship, or promotion claim is made or implied.

## Non-goals of this PR

- No `--policy targeted` implementation (needs a `CoverageGapManifestV1`
  consumer; SLM-265's `iter-slm265-domain-shift-audit-20260724.json` is
  `coverage_gap_manifest/v1`-shaped evidence but is not yet exposed as a
  stable manifest artifact for a generator to consume).
- No `publish_programspec_corpus.py` / `DataStore.publish` — this stays a
  local, inspectable staging corpus under `outputs/data/programspec/`, not a
  durable Git-published dataset.
- No 100k/1M rung, no 3-seed training-arm comparison, no checkpoint.

## Related

- Generator: [`src/slm_training/data/progspec/generate.py`](../../src/slm_training/data/progspec/generate.py)
- New CLI: [`scripts/generate_programspec_corpus.py`](../../scripts/generate_programspec_corpus.py)
- Tests: [`tests/test_scripts/test_generate_programspec_corpus.py`](../../tests/test_scripts/test_generate_programspec_corpus.py)
- SLM-265 evidence consumed for status only (not as a coverage manifest):
  [`iter-slm265-domain-shift-audit-20260724.json`](iter-slm265-domain-shift-audit-20260724.json)
