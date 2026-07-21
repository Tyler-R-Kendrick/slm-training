# SLM-195 (FFE3-04) solver-only semantic ceiling

- **Run ID:** slm195_cli_test
- **Experiment:** slm195-solver-only-semantic-ceiling
- **Matrix set:** slm195_solver_only_semantic_ceiling
- **Matrix version:** ffe3-04-v1
- **Fixture:** vss4-fixture-word/v1
- **Source commit:** aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
- **Random seed:** 0
- **Disposition:** solver_ceiling_established
- **Timestamp:** 2026-07-21T10:57:24.321486Z

## Arms

| arm | budget | status | decisions | terminal | wall (s) |
| --- | --- | --- | --- | --- | --- |
| canonical_dfs | 10 | SOLVED_ACCEPTED | 1 | `aa` | 0.001305 |
| canonical_dfs | 100 | SOLVED_ACCEPTED | 1 | `aa` | 0.001086 |
| canonical_dfs | 1000 | SOLVED_ACCEPTED | 1 | `aa` | 0.000953 |
| random_order | 10 | SOLVED_ACCEPTED | 1 | `aa` | 0.001361 |
| random_order | 100 | SOLVED_ACCEPTED | 1 | `aa` | 0.001334 |
| random_order | 1000 | SOLVED_ACCEPTED | 1 | `aa` | 0.000970 |
| oracle_order | 10 | SOLVED_ACCEPTED | 1 | `aa` | 0.000961 |
| oracle_order | 100 | SOLVED_ACCEPTED | 1 | `aa` | 0.000973 |
| oracle_order | 1000 | SOLVED_ACCEPTED | 1 | `aa` | 0.001027 |
| bfs_min_edits | 10 | SOLVED_ACCEPTED | 3 | `aa` | 0.000976 |
| bfs_min_edits | 100 | SOLVED_ACCEPTED | 3 | `aa` | 0.000990 |
| bfs_min_edits | 1000 | SOLVED_ACCEPTED | 3 | `aa` | 0.001015 |
| astar_admissible | 10 | SOLVED_ACCEPTED | 3 | `aa` | 0.001075 |
| astar_admissible | 100 | SOLVED_ACCEPTED | 3 | `aa` | 0.001010 |
| astar_admissible | 1000 | SOLVED_ACCEPTED | 3 | `aa` | 0.001017 |
| beam_symbolic | 10 | SOLVED_ACCEPTED | 4 | `aa` | 0.001019 |
| beam_symbolic | 100 | SOLVED_ACCEPTED | 4 | `aa` | 0.001020 |
| beam_symbolic | 1000 | SOLVED_ACCEPTED | 4 | `aa` | 0.001006 |
| search_work_energy | 10 | SOLVED_ACCEPTED | 1 | `aa` | 0.001043 |
| search_work_energy | 100 | SOLVED_ACCEPTED | 1 | `aa` | 0.001014 |
| search_work_energy | 1000 | SOLVED_ACCEPTED | 1 | `aa` | 0.001018 |

## Budget grid

### budget=10

- **canonical_dfs**: SOLVED_ACCEPTED (1 decisions)
- **random_order**: SOLVED_ACCEPTED (1 decisions)
- **oracle_order**: SOLVED_ACCEPTED (1 decisions)
- **bfs_min_edits**: SOLVED_ACCEPTED (3 decisions)
- **astar_admissible**: SOLVED_ACCEPTED (3 decisions)
- **beam_symbolic**: SOLVED_ACCEPTED (4 decisions)
- **search_work_energy**: SOLVED_ACCEPTED (1 decisions)

### budget=100

- **canonical_dfs**: SOLVED_ACCEPTED (1 decisions)
- **random_order**: SOLVED_ACCEPTED (1 decisions)
- **oracle_order**: SOLVED_ACCEPTED (1 decisions)
- **bfs_min_edits**: SOLVED_ACCEPTED (3 decisions)
- **astar_admissible**: SOLVED_ACCEPTED (3 decisions)
- **beam_symbolic**: SOLVED_ACCEPTED (4 decisions)
- **search_work_energy**: SOLVED_ACCEPTED (1 decisions)

### budget=1000

- **canonical_dfs**: SOLVED_ACCEPTED (1 decisions)
- **random_order**: SOLVED_ACCEPTED (1 decisions)
- **oracle_order**: SOLVED_ACCEPTED (1 decisions)
- **bfs_min_edits**: SOLVED_ACCEPTED (3 decisions)
- **astar_admissible**: SOLVED_ACCEPTED (3 decisions)
- **beam_symbolic**: SOLVED_ACCEPTED (4 decisions)
- **search_work_energy**: SOLVED_ACCEPTED (1 decisions)

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, GPU, or ship-gate claim is involved.
- The reference fixture is the committed VSS finite word tree; the only verifier-accepted terminal is 'aa'.
- Symbolic rankers (including search_work_energy) are deterministic stand-ins for a learned energy model.
- A* and beam heuristics are admissible w.r.t. remaining decisions, not a learned value function.
- Random-order results depend on the manifest random_seed and are expected to vary.

## Interpretation

This harness establishes a solver-only ceiling on the exact VSS finite fixture. Any learned model that claims to improve on these symbolic baselines must be evaluated against the same fixture and proven not to introduce false UNSAT or unsupported candidate deletions.
