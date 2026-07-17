# E260/E261 — C4 head-on test of the "names disappear" threat (2026-07-17)

Fixture-grade matched pair for Track C4 (Linear SLM-28). Machine-readable
evidence:
[quality-matrix-results-iter-v13-c4-20260717.json](quality-matrix-results-iter-v13-c4-20260717.json).
Code: `symbol_anonymization` flag in
[`src/slm_training/models/dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py)
(encode) and
[`src/slm_training/models/twotower.py`](../../src/slm_training/models/twotower.py)
(config + fail-closed guards).

## Why this exists

"When Names Disappear" ([arXiv:2510.03178](https://arxiv.org/abs/2510.03178))
plus Gao 2023 / Yang 2022 show identifier anonymization substantially degrades
code models. The program's C-track (C1 relative refs, C2 pseudo-embeddings,
C3 macros) is built on the defense that OpenUI content is placeholder-routed,
so identifier identity carries no semantics here. C4 tests that defense
empirically instead of assuming it.

## The controlled pair

One lever: `symbol_anonymization`. The surface arm (E261) encodes binder and
state names verbatim through the existing byte channel; the control (E260)
keeps the standard `<BIND_j>`/`<STATE_k>` pools. Placeholders ride `<SYM_i>`
in **both** arms (the placeholder lever was already tested and rejected as
E49), so the comparison isolates exactly the identifier-anonymization choice
the paper attacks. Round trip in surface mode is exact without a symbol table
(verified over all 108 fixture corpus records; corpus cost 1.72× tokens:
8,898 vs 5,180).

Both arms decode **unconstrained** (`grammar_constrained=False`, new
per-experiment knob): the fastpath NAME gate admits only `<BIND_j>` ids, so
constrained decode could never emit a byte-spelled identifier — running the
gate would confound the representation lever with a decode-legality artifact.
Fail-closed guards refuse surface mode combined with `grammar_constrained`,
`macro_tokens`, or `bind_encoding="relative"` (all presuppose pooled ids).

## Fixture result (wiring evidence only)

Recipe: `--matrix v13 --steps 80 --device cpu --context-backend scratch
--no-design-md-context --rico-limit 3 --scratch-control`, batch 4, seed 0,
lr 3e-4, fixture v1 corpus (108 records); suites smoke 3 / held_out 5 /
adversarial 4 / ood 4 / rico_held 3.

| Metric | E260 anonymized | E261 surface |
| --- | --- | --- |
| syntax parse (all suites) | 0.0 | 0.0 |
| meaningful parse (all suites) | 0.0 | 0.0 |
| structural similarity (sm/ho/adv/ood/rico) | 0.12 / 0.09 / 0.11 / 0.09 / 0.03 | **0.23 / 0.17 / 0.16 / 0.18 / 0.11** |
| train loss @80 | 10.02 | 8.29 |
| seen_target_tokens | 15,417 | 26,489 |
| AgentV | 0/5 | 0/5 |

Losses are **not** directly comparable across arms (different target token
distributions). The comparable secondary signal is structural similarity,
where the surface arm is ahead on every suite despite spending the same 80
steps on 1.72× longer targets.

## Verdict (explicit, as the issue requires)

**Open — cannot confirm or refute at this budget, with a weak fixture-scale
signal consistent with the threat.** The primary metric (meaningful parse) is
0.0 on both arms — the valid-but-empty wall dominates both, so the comparison
never reaches the regime where the paper's effect would be measurable. The
directionally consistent structural-similarity gap (surface > anonymized on
5/5 suites) is exactly what arXiv:2510.03178 predicts, but n ≤ 5 per suite,
one seed, 80 CPU steps — this is a hypothesis-preserving observation, not a
result. The decisive test is a frontier-scale replicated pair (multiple
seeds, real budget, full suites). Until that runs, **the C1–C3 anonymization
defense remains an assumption, now with a small adverse data point rather
than none.**

## Verification

- `tests/test_harnesses/model_build/test_dsl_tokenizer.py::test_surface_identifiers_round_trip_and_isolate_the_lever`
  (exact round trip on three programs incl. state/builtin surface, no
  BIND/STATE ids emitted, placeholders still SYM, relative-encoding rejection).
- `tests/test_harnesses/model_build/test_lexer_smoke.py::test_surface_identifier_arm_trains_and_fails_closed`
  (end-to-end build/train on the surface arm; ValueError on the three
  incompatible combinations).
- 108/108 fixture-corpus round trips exact; full pass over `tests/test_data`,
  `tests/test_harnesses/model_build`, `tests/test_models`, `tests/test_scripts`;
  `repo_policy`, `ruff`, `git diff --check` clean.

## Honesty and limits

- Fixture/scratch wiring evidence only; no ship claim, no gate weakened,
  nothing promoted. The E-row verdict field stays open.
- Removing the grammar gate from both arms changes the decode regime relative
  to E255/E259-style rows; E260 exists precisely so the pair stays internally
  matched. Cross-matrix comparisons to constrained rows are not valid.
- The surface arm is a measurement instrument, not a candidate
  representation: it forfeits the fixed-vocab NAME gate, macro channel, and
  relative binding. If the frontier-scale run confirms the threat, the
  program response would be richer symbol-token features (C2 direction), not
  a wholesale return to surface identifiers.
