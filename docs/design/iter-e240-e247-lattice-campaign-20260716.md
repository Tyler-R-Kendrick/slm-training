# E240–E247 — V9 lattice-guided recursive compiler search campaign (2026-07-16)

Fixture-grade execution of the previously plan-only V9 campaign
([`lattice-recursive-search.md`](lattice-recursive-search.md)). Machine-readable
evidence:
[quality-matrix-results-iter-v9-lattice-20260716.json](quality-matrix-results-iter-v9-lattice-20260716.json).
Linear SLM-21.

## What ran

All eight registered rows executed through `scripts/run_quality_matrix.py`
(`--matrix v9`) on CPU against the standard honest five-suite scoreboard with
AgentV publication per row:

- **E240** (greedy compiler-tree control) trained as an explicit
  `--scratch-control` row: 800 steps, batch 4, seed 0, scratch context backend,
  fixture v1 corpus (108 records, `--source fixture`), no DESIGN.md context.
- **E241–E247** ran **eval-only from E240's frozen checkpoint** through the new
  `--eval-checkpoint` flag, so every row compares decode policy over identical
  checkpoint/data lineage, as the campaign spec requires.

`--eval-checkpoint` is new in this iteration: the V9 rows are registered with
`initialization="eval_only"`, but the matrix classifier silently reclassified
them to `parent`/`scratch` (each row would have trained its own model,
violating matched lineage). The flag routes declared eval-only rows through one
frozen checkpoint; regression tests cover both the declaration and the routing
(`tests/test_scripts/test_quality_matrix_v9.py`).

Suite sizes: smoke 3, held_out 5, adversarial 4, ood 4, **rico_held 0** (the
fixture corpus contains no RICO records; `--rico-limit 3` recorded, full 1500
remains the ship bar).

## Scoreboard (all rows fail honest gates — expected at fixture scale)

Every row scores syntax parse 0.0 / meaningful parse 0.0: the 800-step scratch
model emits free-form string literals (`"fo"`, `"s"`) where the placeholder
policy requires `:scope.name` references, so even raw syntax validity fails.
This is a genuine fixture-scale capacity result (bridge verified working — it
returns per-record placeholder-policy errors, not zeros from a broken
verifier). Structural similarity is nonzero (0.24–0.41), so decode produces
real structure through the compiler tree.

## Lattice diagnostics (the campaign's actual signal)

Sums across smoke/held_out/adversarial/ood; ms/rec is mean decode wall per
record averaged over suites.

| ID | Mode | Lattice states | Candidates | Traj. triggers | Trajectories | Unique proposals | Bottoms/rollbacks/nogoods | ms/rec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E240 | greedy | 0 | 0 | 0 | 0 | 0 | 0 | 9060 |
| E241 | lattice | 891 | 29089 | 0 | 0 | 0 | 0 | 8108 |
| E242 | lattice+nogood | 891 | 29089 | 0 | 0 | 0 | 0 | 7927 |
| E243 | ptrm, stagnation w4 | 891 | 29089 | 0 | 0 | 0 | 0 | 7881 |
| E244 | ptrm, always w4 | 1696 | 146240 | 1680 | 6640 | 6544 | 0 | 24113 |
| E245 | gram, stagnation w4 | 891 | 29089 | 0 | 0 | 0 | 0 | 6253 |
| E246 | full, stagnation w4 | 891 | 29089 | 0 | 0 | 0 | 0 | 6274 |
| E247 | full, stagnation w8 | 891 | 29089 | 0 | 0 | 0 | 0 | 6213 |

Observations against the registered falsifiers:

1. **E240 control reproduces existing valid decode behavior** — the greedy
   compiler-tree path runs end-to-end and its outputs are byte-identical to
   every stagnation-triggered row. Derivative hypothesis 1 ("projection before
   preference does not change the default greedy result on singleton/complete
   forests") is directly observed on this checkpoint.
2. **The hard/soft lattice observes the forest without distorting it**
   (E241–E243, E245–E247): 891 lattice states and ~29k ranked candidates per
   suite sweep, zero bottoms, zero stagnation events — so rollback, nogoods,
   and triggered trajectories never fire on this checkpoint. Their scoreboards
   and latency match greedy.
3. **Always-on PTRM (E244) is strictly worse than triggered search here**:
   1680 trigger events / 6640 seeded trajectories / 6544 unique proposals cost
   ~3× wall latency (24.1s vs ~8s per record) and *reduce* structural
   similarity on every suite (smoke 0.353→0.323, held_out 0.315→0.246,
   adversarial 0.377→0.300, ood 0.413→0.239). At fixture scale this supports
   the "selective stochasticity" hypothesis over always-on perturbation and
   satisfies E244's role as the matched control.

## Honesty and limits

- **Wiring evidence only, not a ship claim.** The checkpoint is an 800-step
  CPU scratch control on the 108-record fixture corpus; no gate is weakened,
  nothing is promoted, and rows failing honest gates is the expected outcome.
- **The rollback/nogood/bottom machinery never triggered** on this checkpoint
  (greedy decode never stalls), so E241/E242's runtime conflict behavior is
  exercised only by unit/integration tests
  (`tests/test_dsl/test_lattice_search.py`,
  `tests/test_models/test_compiler_decode.py`), not by this campaign.
- **The ship-grade campaign remains open**: the same invocation against the
  local E224+ frontier checkpoints (gitignored, GPU host) with full suites
  (`rico_held` n=1500) is required before any claim about whether lattice
  search moves the valid-but-empty wall (Track A comparison baseline for
  A2–A4).
- GRAM semantic dedup at decode time currently deduplicates by first-token
  proposal identity; the AST-fingerprint dedup utility exists
  (`dsl/grammar/fastpath/lattice_search.py`) but is not yet wired into the
  runtime loop — E245's unique-valid-AST falsifier can only be settled after
  that wiring (tracked as follow-up).
