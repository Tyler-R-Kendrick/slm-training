# EXP-SR-9: prospective certified macro/library-learning experiment

Campaign manifest: `578327e04b402f210aa92e6188210ef4af6c4bc92189e97db58129d978002872` (claim_class=`fixture`).
Catalogue manifest: `31af54bdc22065423ee2b54339f0eefa133226c67d3ea8265edbf3b330153db8` (`exp-sr-9`).

Fixture-scale certified macro/library-learning campaign over a small deterministic in-process symbolic-regression corpus (symbolic_expr_corpus.generate_corpus). Macros are closed (0-arity) recurring canonical subtrees mined exclusively from corpus_split=='train' records and evaluated on the frozen corpus_split=='test' slice, never the reverse. claim_class=fixture; 'authorized' is a diagnostic signal for later structural-search work and never promotes a checkpoint.

- Train records: 64
- Held-out (frozen, never mined from) test records: 8
- Candidate subtrees observed while mining: 205

## Arms (prospective `macro_library_size_reduction_rate` on the held-out slice)

| Arm | Reduction rate | Library size | Selection policy |
| --- | --- | --- | --- |
| `no_macros` | 0.000000 | 0 | `none` |
| `frequency_macros` | 0.194175 | 16 | `frequency_only` |
| `mdl_macros` | 0.194175 | 16 | `mdl_net_gain` |

- Primary metric: `macro_library_size_reduction_rate`, minimum_effect=0.02
- Leakage audit: `is_clean`=True
- Expansion-equivalence: 16 admitted mdl_macros checked, all_verified=True
- Falsifier holds: False

**Authorized (diagnostic signal only, never a promotion):** `True`

Full detail: `docs/design/exp-sr-9-macro-library-experiment.json`.
