# Witness-prune honesty: UNKNOWN ≠ UNSUPPORTED (2026-08-04)

Advisory instrumentation, no behavior change, outputs byte-identical.
Triggered by adversarial review of an external "visibility determination"
analysis of the decoder. Claim class: diagnostic/fixture. Companions:
[`hybrid-unmask-hx5-20260804.md`](hybrid-unmask-hx5-20260804.md),
[`residual-honesty-block-diffusion-20260804.md`](residual-honesty-block-diffusion-20260804.md).

## The finding

The external analysis praised this repo's three-valued support logic
(`SUPPORTED` / `UNSUPPORTED` / `UNKNOWN`, where "UNKNOWN never licenses
candidate removal") as its core soundness advantage over mainstream
constrained decoders. **That law holds only in `dsl/solver/`** — a different
subsystem with a different enum (`SupportVerdict`, `UNKNOWN_POLICIES =
("keep_and_rank",)`).

The production completion-domain path uses `WitnessStatus` and **violates
it**. In `dsl/pack.py` the candidate is dropped identically for both
verdicts:

```python
proof = session.terminal_witness(child, budget - len(tokens))
if proof.status is WitnessStatus.UNKNOWN:
    unsupported = True      # <- budget exhausted, NOT proven impossible
    continue
if proof.status is WitnessStatus.UNSUPPORTED:
    unsupported = True      # <- certified impossible
    continue
```

Both collapse into a single output `reason="witness_pruned"`, so callers
cannot distinguish *certified impossible* from *we ran out of search
budget*. Dropping on UNKNOWN is a **false reject**: it silently narrows the
legal domain (distinct from widening it — this direction can make valid
programs unreachable rather than admitting invalid ones). The in-code
comment concedes the behavior ("an unproven candidate is omitted") and
defends it as V1 bug-compatibility, not soundness. A second path
(`_tail_from` returning `None`) has the same shape.

## What the instrumentation measured

New advisory counters (`witness_materialized`, `witness_kept`,
`witness_pruned_unknown`, `witness_pruned_unsupported`), profile fixture
(`outputs/runs/s1_d64`, `--maskgit --rounds 2`, 6 generates, idle machine),
outputs byte-identical to the pre-instrumentation run:

| counter | value |
| --- | --- |
| `witness_materialized` | 2180 |
| `witness_kept` | 2104 (**96.5%**) |
| `witness_pruned_unknown` | 88 (**4.0%**) |
| `witness_pruned_unsupported` | **0** |
| `constrained_dead_ends` | **0** |
| `certified_fallbacks` | 6 |

### Three conclusions, two of which kill proposed work

1. **The soundness gap is real but small, and it is entirely one-sided.**
   Every witness-based rejection on this fixture is the *unsound* kind: 88
   UNKNOWN false-rejects, **zero** certified UNSUPPORTED. The
   soundness-preserving branch is dead code here — the mechanism never once
   made a proven rejection. The gap is worth fixing on correctness grounds,
   and the fix (keep UNKNOWN candidates and rank them last, mirroring
   `dsl/solver`'s existing `keep_and_rank` policy) is cheap.

2. **The causal hypothesis is NOT supported.** I hypothesised that
   UNKNOWN-dropped legal candidates drive constrained dead ends → LTR repair
   → the measured ~77% finalize cost. `constrained_dead_ends = 0` on this
   fixture: the 88 false rejects caused no dead ends, so they do **not**
   explain the repair cost. Honest negative; the finalize bottleneck remains
   unexplained by this mechanism.

3. **Lazy witness materialization is worth ≤3.5%, not a transformative
   win.** The external analysis's flagship proposal (VIS-H2: separate
   candidate membership from eager witness construction) can only save work
   on witnesses that are ultimately *discarded* — and **96.5% are kept**. One
   counter, one fixture run, sized a large proposed redesign out of
   contention before any of it was built.

## `speculative_overlap` — default-off finally justified

The repo ships a `speculative_overlap` lever that overlaps grammar
verification with the denoiser forward (`ThreadPoolExecutor`, one worker),
default off, with **no recorded measurement** justifying the default. It
only takes effect inside the cluster-verify path, so a naive on/off A/B
outside that path measures nothing.

Isolated A/B (cluster mode + `cluster_verify`, 3 repetitions each, medians,
identical outputs across all arms):

| arm | walls (s) | median |
| --- | --- | --- |
| overlap OFF | 3.29 / 2.99 / 3.54 | 3.29 |
| overlap ON | 3.40 / 2.93 / 3.32 | 3.32 |

**+0.9% — inside the declared ±10% noise band.** No measurable benefit on
this fixture; the default-off is now justified in writing rather than by
omission. This is expected: the overlapped grammar work is CPU-bound Python
racing a CPU-bound denoiser on the same cores, so there is no idle resource
to reclaim. It would need a GPU denoiser (the configuration XGrammar's
overlap result assumes) to have a chance.

## Corrections to the record

- **Vocabulary is 569, not 296.** Two design docs
  (`dsl-native-tokenizer.md`, `quality-experiment-matrix.md`) claimed a
  "fixed output vocab **296**" and **misled an external reviewer into a
  wrong architectural inference**. The authority is
  `resources/tokenizer_layout_registry.json` → `vocab_size: 569`,
  `version: 5`, `layout_sha256: 381a05d0…`; `DSLNativeTokenizer.build()`
  yields 569 = 132 lit + 96 byte + 64 sym + 64 bind + 64 state + 64 macro +
  35 component + 26 struct + 19 builtin + 5 special. Both docs corrected
  with a historical note. (`agent-harness-parity-audit.md` had already
  caught the contradiction; the stale docs were never updated.)
- **Finalize share is ~77%, not 87%.** The 87% figure circulating in the
  campaign docs came from CPU-contended arms; the idle `origin/main`
  measurement is 3308.9 / 4274.7 = **77.4%**. Same provenance defect as the
  40% HX5 speedup already retracted in the hybrid doc. Quote 77%, idle-only.

## Honesty

Fixture-scale evidence: one local checkpoint, one prompt for the overlap
A/B, 6 generates for the counters, single machine. `witness_pruned_unknown`
being 100% of prunes is a property of *this* fixture — a harder corpus could
produce certified UNSUPPORTED rejections. No behavior changed; no quality
claim; no lever default flipped. The counters are advisory and ungated.

## Follow-ups

- **HV-B (recommended next)**: keep UNKNOWN candidates and rank them last
  behind a default-off lever, verified against the exact oracle — this
  deliberately *widens* the emitted domain toward the true legal set, so it
  must be checked for admissibility, not merely accepted.
- Raise the witness `node_budget` and re-measure the 88 UNKNOWN drops; if
  they vanish cheaply, the false-reject risk closes without new machinery.
- VIS-H2 (lazy witnesses) is **not recommended** on this evidence (≤3.5%
  ceiling); revisit only if a corpus shows a materially lower keep rate.
- The competitive-benchmark ask (vs XGrammar/llguidance/Outlines) is already
  preregistered as **H17 / SDE3-04** with a frozen manifest and
  `activation_status: activation_blocked`, `max_dollars: 0.0`. The action is
  to clear `eval_cache_or_cost_approved` + `budget_approved` and amend the
  manifest, not to build a harness — and a 10k-schema sweep cannot run under
  `MAX_RUN_MINUTES = 3` without sharding into durable per-chunk artifacts.
