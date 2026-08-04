# False-singleton guard + shard makespan fix (2026-08-04)

Two faults fixed, one behavior change deliberately **not** made. Claim class:
diagnostic/fixture. Follows
[`witness-prune-honesty-20260804.md`](witness-prune-honesty-20260804.md).

## Fault B — CI shard packing optimised the wrong objective (regression, mine)

The duration-aware packing landed earlier that day kept each **measured file
whole**, on the theory that a known-slow suite should be isolated. That is the
wrong objective: the canonical per-job wall (`MAX_RUN_MINUTES`) is enforced
**per shard**, so a packing is good exactly when its *slowest* shard fits.
Keeping files atomic makes the worst shard equal to the single heaviest file.

Simulated over the committed weights table (12 shards, real durations,
**after** the `test_topology_apply.py` weight correction described below):

| packing | worst shard | ideal |
| --- | --- | --- |
| atomic files (previous) | **4726.2 s** | 486.5 s |
| per-node (this fix) | **1575.4 s** | 486.5 s |
| per-node **+ slow files deselected** (actual CI selection) | **67.6 s** | 67.6 s |

Per-node packing is 3× better than atomic, but on its own it still does not
fit the wall — the two irreducible files dominate. Only the combination of
per-node packing *and* deselecting those files reaches a worst shard that
fits (67.6 s against a ~100 s post-setup budget), and it lands exactly on the
ideal. (An earlier version of this table read 429.1 / 143.0 / ideal 128.5,
computed from the wrong `test_topology_apply.py` weight; the ordering of the
policies was right but every absolute number was understated ~11×.) `_shard_test_nodes` now spreads a measured file's
weight across its own nodes; an unmeasured file still contributes one unit per
node, so an empty table still reduces exactly to count-balanced round-robin.
The test that asserted heavy-file *isolation* encoded the wrong goal and was
replaced by one asserting **makespan** plus the anti-isolation property (a
heavy file's nodes must spread).

### The irreducible remainder

Per-node splitting cannot help a file whose weight sits in **one test node**.
Measured on an idle machine:

| file | nodes | measured (idle) |
| --- | ---: | ---: |
| `test_dsl/test_topology_apply.py` | 14 | **4726 s** (~338 s/node) |
| `test_scripts/test_run_dsh5_03_bulk_operator_crossover.py` | **1** | **252 s** |

> **Correction (same day).** This table first read *"429 s (~31 s/node —
> splits fine)"* for `test_topology_apply.py`, taken from the committed
> weights table. That entry was a **partial measurement** — the generating
> sweep was killed while the file was still running, a caveat its own risk
> note recorded — and it is wrong by **11×**. Run to completion on an idle
> machine the file takes **4726 s**. Per-class breakdown:
> `TestEditKindMapping` 3.0 s (3 tests), `TestDefaultOff` 309.9 s (3 tests),
> `TestStaleInvalidation` >600 s (6 tests), `TestDirectApply` the residual.
> Individual nodes run ~100–340 s, so **per-node splitting cannot save this
> file either** — the original claim that it "splits fine" was false.
> The weights table is corrected to 4726.16 and the three heavy classes are
> now `@pytest.mark.slow`; `TestEditKindMapping` stays in CI, where the file
> now completes in 0.11 s instead of being cancelled at the wall.

The `dsh5_03` case is a single test whose cost is inside `run_local_preflight()`
(import is only 3.7 s), so no packing scheme can fit it. It has been
**cancelled on every CI run** — producing no evidence while reddening a shard.
It is now marked `@pytest.mark.slow` and deselected by default
(`addopts = "-m 'not training and not slow'"`), which converts a silent
timeout into a documented deferral. Run explicitly:

```bash
python -m pytest tests/test_scripts/test_run_dsh5_03_bulk_operator_crossover.py -m slow
```

**This is a coverage reduction in CI and is stated as such.** The follow-up is
to make `run_local_preflight` cheaper or split the test, not to leave it
deselected forever.

## Fault A — UNKNOWN false rejects: guarded, not "fixed"

The completion-domain filter drops a candidate for a budget-exhausted
`UNKNOWN` witness exactly as it does for a certified `UNSUPPORTED` one (see
the companion doc). Four measurements now bound the fault:

| question | measurement |
| --- | --- |
| How often does it happen? | 88 of 2180 materialized witnesses (4.0%); **zero** certified UNSUPPORTED |
| Does it cause dead ends? | `constrained_dead_ends = 0` |
| Does a bigger budget fix it? | budget 16→64 recovers 2 of 10 drops at **2.4× wall**; 64→256 recovers **zero** more |
| Does it manufacture false singletons? | **`witness_false_singleton_risk = 0`** |

The last row is the sharp hazard and the reason for the new counter. If an
UNKNOWN drop leaves exactly **one** proven survivor, that survivor looks
deterministically forced to `exact_forced_token_id`, which commits it with
**zero model forwards** under I2 — a bypass justified by a search budget
rather than a proof. The counter flags that shape. It reads zero on this
fixture, and the detector is proven live by a unit test (a zero counter is
only evidence if it can fire).

### Why no behavior change

The obvious fix — keep UNKNOWN candidates and rank them last, mirroring
`dsl/solver`'s existing `keep_and_rank` policy — is **not** implemented here,
because every measurable consequence of the fault is currently nil: no dead
ends, no false singletons, no invalid outputs, and raising the budget is
strictly expensive. Shipping a behavior-changing lever with no measured
benefit is exactly the pattern this line of work has already produced three
times (block-diffusion lever slower, hybrid slower, HX5 neutral).

The fault is real and stays on the record. What would justify the fix:
`witness_false_singleton_risk > 0`, or a corpus where UNKNOWN drops correlate
with dead ends or certified fallbacks. The counter now makes that
self-reporting: if the hazard ever occurs, it is visible rather than silent.

## Verification

144 tests green (packs, completion kernel, E1 static-control-domain, language
contract, decode stats, check_changed, repo policy) plus the new counter
tests. Decode profile outputs **byte-identical**, with the witness counters
reproducing the merged baseline exactly (2180 / 2104 / 88) — proof the added
tally changed nothing.

`src/slm_training/dsl/pack.py` was re-pinned in the DSH0-02 evidence record
again, with the acceptance criteria re-exercised, per the rationale recorded
there.

## Honesty

Fixture-scale, one checkpoint, one prompt for the budget probe. The zero
false-singleton count is a property of this fixture, not a proof that the
shape cannot occur. The `slow` marker removes a test from default CI runs.
The shard simulation assumes 3 nodes per measured file for unmeasured
granularity; real node counts were used for the heavy files named above.
