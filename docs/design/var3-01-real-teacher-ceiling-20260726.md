# VAR3-01: real teacher behind OperatorTeacherAdapter, DSH4-02 stop rule re-read (SLM-429)

**Status:** experiment complete (with one honestly-recorded environment caveat).
**Claim class:** `experiment`.
**Honest verdict:** `no_go_defer` -- the stop rule fired on the **real**
teacher, exactly as it fired on the synthetic stand-in.
**Closes:** the three DSH4-02 preconditions for a production KD-readiness
*read* (not a readiness claim -- the read is a no-go).

DSH4-02 (`dsh4-02-operator-teacher-ceiling-fixture-20260725.md`) deferred
because no real teacher was wired -- only the deterministic
`SyntheticDescriptorTeacherV1` stand-in. This iteration wires
`PromptedTeacherV1`, a real prompted LLM
(`Qwen/Qwen3-4B-Instruct-2507`, temperature 0, OpenAI-compatible HF router
endpoint), behind the **existing, unmodified** `OperatorTeacherAdapter`
Protocol, re-runs the **unchanged** `compare_operator_teacher_ceiling`
harness over the **identical** fixture trace pools (same `GOLD_WEIGHTS`,
same seed 42, same 30/30 train/eval split -- the latent gold-label
distribution was fixed before the run and never touched after), and reads
`evaluate_stop_rule` honestly over the real output.
`SyntheticDescriptorTeacherV1` is kept as a control arm.

## What changed

* `src/slm_training/harnesses/distill/prompted_teacher.py` (new) --
  `PromptedTeacherV1`:
  * Implements `OperatorTeacherAdapter` (`source_id`, `rank(query)`).
    **No harness change was needed** -- the Protocol was sufficient, so
    there is no adapter-modification finding to report.
  * Anti-leak surface identical to the synthetic stand-in: the prompt is
    built only from `operator_id`, argument arity, `proof_checks`, and a
    `semantic_id`-derived identity term (16-hex-char content-identity
    prefix), plus the presentation-only `TeacherQueryV1` fields
    (candidate order, opaque presentation labels, description-length
    bounds, prompt-template hash) that are exactly the preregistered
    perturbation surface. It never reads or renders `application_id` /
    opaque refs, `trace.current_scores`, or
    `trace.accepted_application_ids`; unit tests assert the rendered
    prompt cannot contain them.
  * Content-addressed response cache (`cache_dir/<sha256(request)>.json`
    storing request + raw response): re-running the comparison never
    re-queries the provider. Requests are deterministic
    (`temperature=0`).
  * Fail closed: provider errors, unparseable replies, replies whose
    labels do not exactly cover the presented candidate set, and
    non-positive scores all yield `None` (the harness then reports
    insufficient data, never a win). Uncertified text is never returned.
  * Transport is injectable so tests run offline against recorded
    responses.
* `scripts/run_var3_01_real_teacher_ceiling.py` (new) -- runner: rebuilds
  the exact DSH4-02 fixture pools by **importing** the DSH4-02 fixture
  builders and `GOLD_WEIGHTS` (reuse, not duplication), warms the cache
  for every query the unchanged harness will issue (canonical + the four
  perturbation kinds, same seed arithmetic as
  `run_perturbation_robustness`), then runs
  `compare_operator_teacher_ceiling` twice (real teacher; synthetic
  control).
* `tests/test_harnesses/distill/test_prompted_teacher.py` (new) -- 21
  tests: adapter conformance, anti-leak assertions, fail-closed paths,
  content-addressed caching (repeat query never re-queries; cache shared
  across instances), perturbation-surface behavior (labels map back to
  application ids; presentation reaches the prompt; template hash selects
  among preregistered paraphrases), and an end-to-end run of the
  unchanged harness against the prompted teacher with a recorded
  transport.
* `src/slm_training/resources/versions.json` -- `harness.distill` v7 ->
  v8, `harness.experiments` v133 -> v134 (history notes on top).

## DSH4-02 preconditions

| Precondition | Status |
| --- | --- |
| (1) A real teacher behind `OperatorTeacherAdapter` | **Met** -- `PromptedTeacherV1`, Qwen3-4B-Instruct-2507, temperature 0, content-addressed cache. |
| (2) Re-run the same harness against it | **Met** -- `compare_operator_teacher_ceiling` unmodified; identical fixture pools; all baselines, oracle upper bound, both populated `DecisionFamily` slices (this fixture only populates `gold_large` / `on_policy_large` -- all 8 are still sliced in code), all 4 perturbation kinds, paired bootstrap (2000 resamples). See the caveat below on perturbation coverage. |
| (3) An honest read of the stop rule over the real output | **Met** -- `no_go_defer`, reported below without adjustment. |

## Result (honest, unmodified harness output)

```bash
python -m scripts.run_var3_01_real_teacher_ceiling --run-id var3-01-20260726
```

Evidence: [var3-01-real-teacher-ceiling-20260726.json](var3-01-real-teacher-ceiling-20260726.json)
(machine payload incl. `version_stamp`). Output:
`outputs/runs/var3-01-real-teacher-ceiling/var3-01-20260726/`
(`summary.json`, `teacher_cache/` with the content-addressed provider
exchanges).

### Overall metrics (30 eval traces, 5 legal actions each)

| Comparator | MRR | top-3 recall | NDCG@3 | accepted-set mass | calibration ECE | selective-risk AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `oracle:accepted_set` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `baseline:current_scorer` | 0.594 | 0.767 | 0.601 | 0.216 | 0.037 | 0.577 |
| `baseline:frequency` | 0.589 | 0.633 | 0.538 | 0.241 | 0.011 | 0.556 |
| `baseline:descriptor_similarity` | 0.431 | 0.533 | 0.377 | 0.200 | 0.000 | 0.937 |
| `baseline:compiler_order` | 0.418 | 0.767 | 0.469 | n/a (no scores) | n/a | n/a |
| **`teacher:prompted_qwen3_4b_v1`** | **0.418** | 0.767 | 0.469 | 0.201 | 0.020 | 0.963 |
| control `teacher:synthetic_descriptor_v1` | 0.343 | 0.467 | 0.280 | 0.197 | 0.063 | 0.963 |

### Paired MRR, real teacher vs. each required baseline (2000-resample bootstrap, n=30)

| Baseline | mean diff | 95% CI | significantly better? |
| --- | ---: | --- | :---: |
| `baseline:frequency` | -0.171 | [-0.292, -0.031] | no |
| `baseline:compiler_order` | +0.000 | [+0.000, +0.000] | no (exact tie) |
| `baseline:current_scorer` | -0.176 | [-0.295, -0.041] | no |
| `baseline:descriptor_similarity` | -0.012 | [-0.147, +0.114] | no |

### Perturbation robustness (real teacher)

| Kind | max ranking distance | bound | n traces compared | within bound |
| --- | ---: | ---: | ---: | :---: |
| candidate order | 0.600 | 0.000 | 16 | **no** |
| opaque-id relabel | 0.000 | 0.000 | 5 | yes |
| description length | 0.600 | 0.000 | 6 | **no** |
| prompt-template hash | 0.000 | 0.000 | 22 | yes |

Per the SLM-429 acceptance bar, a teacher whose ranking moves under
perturbation is **reported as unreliable regardless of its accuracy** --
this one moves under candidate-order shuffle and description-length
variation.

### Verifier-regret correlation (pooled Spearman, probability vs. accepted)

| Comparator | correlation |
| --- | ---: |
| `oracle:accepted_set` | 1.000 |
| `baseline:current_scorer` | 0.197 |
| `baseline:frequency` | 0.137 |
| **`teacher:prompted_qwen3_4b_v1`** | **0.059** |
| control `teacher:synthetic_descriptor_v1` | -0.153 |

### Stop rule (real teacher)

```json
{
  "go": false,
  "recommendation": "no_go_defer",
  "beats_required_baselines": false,
  "failing_baselines": [
    "baseline:compiler_order", "baseline:current_scorer",
    "baseline:descriptor_similarity", "baseline:frequency"
  ],
  "perturbation_within_bound": false,
  "correlation_positive": true,
  "verifier_regret_correlation": 0.05892556509887897
}
```

The control arm (`SyntheticDescriptorTeacherV1`) also reads `no_go_defer`
(fails all four baselines; correlation -0.153; perturbation within bound).

**Honest verdict: DEFER. Do not distill this teacher.** The real teacher
fails two of the three gate criteria: it does not significantly beat any
required baseline on paired MRR (it *significantly loses* to frequency and
the current scorer and exactly ties compiler order), and its ranking is
fragile to candidate-order and description-length perturbation (max
distance 0.6 against a declared bound of 0.0). Its one passing criterion --
a positive but weak verifier-regret correlation (+0.059) -- does not
offset the other two.

## What the real teacher actually did (why the numbers look like this)

The teacher's canonical-run MRR is *bit-identical* to the compiler-order
baseline's (paired diff exactly +0.000 on all 30 traces): on this fixture
the model ranked candidates in their presented order every time. The
fixture presents five candidates of a single-argument operator whose only
legitimate difference is a 16-hex-char content-identity term -- there is no
natural-language meaning for the model to latch onto, so it falls back to
presentation order, which is precisely the fragility the perturbation
tests exist to catch (candidate-order shuffle moves its ranking by 0.6).
This is the harness working as designed on a real teacher: an LLM given
only structural digests has no information advantage over compiler order,
and its apparent "signal" is presentation echo. That closes this
*approach* (prompted ranking over digest-level candidate descriptions), not
the goal: the documented successor approach is a teacher query with
realization-level but still leak-free candidate descriptions (e.g.
DSH3-06-certified, request-local value descriptors rather than bare
identity prefixes), which is a data-visibility change to the *query
builder*, not to the harness or the stop rule.

## Honest caveats

* **Environment caveat (provider credits):** mid-run the HF Inference
  Providers account returned HTTP 402 (monthly included credits
  depleted). All 30 canonical rankings -- every number in the paired
  comparison, metrics tables, family slices, and the stop-rule baseline
  criterion -- were completed **before** exhaustion and are fully cached
  (79 content-addressed exchanges in `teacher_cache/`). The perturbation
  kinds were measured on the subsets shown above (candidate order n=16,
  opaque-id n=5, description length n=6, prompt template n=22). The
  verdict cannot flip on the missing queries: the perturbation bound
  (0.0) is already exceeded at 0.6, and the baseline criterion is
  computed from the complete canonical set. Re-running the script with
  provider credit fills the remaining exchanges from the same cache keys
  without re-querying the completed ones.
* **Model substitution recorded honestly:** the run originally targeted
  `Qwen/Qwen2.5-7B-Instruct`; the router stopped serving it mid-session
  (`model_not_supported`), so the teacher was pinned to
  `Qwen/Qwen3-4B-Instruct-2507` -- recorded here and in the payload's
  `teacher_model` field. This is a provider-availability substitution of
  the teacher *model*, made before any results were seen, not a tuning of
  gold labels, the stop rule, families, or perturbations.
* **Fixture scope unchanged:** single-operator fixture; candidates differ
  only in content identity. The result is a statement about *this*
  teacher on *this* visibility surface, not about LLM teachers in
  general -- see the successor approach above.
* **No distillation follows.** Per the stop rule and SLM-429's honest-
  verdict requirement, `no_go_defer` against a real teacher closes an
  approach (I14), not the goal.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_prompted_teacher.py -q   # 21 passed
python -m pytest tests/test_harnesses/distill/test_operator_teacher_ceiling.py -q  # 39 passed
python -m scripts.run_var3_01_real_teacher_ceiling --run-id var3-01-20260726
ruff check src/slm_training/harnesses/distill/prompted_teacher.py \
  scripts/run_var3_01_real_teacher_ceiling.py \
  tests/test_harnesses/distill/test_prompted_teacher.py
python -m scripts.verify_version_stamps --check
python -m scripts.repo_policy
```

All passed on this branch at the time of writing. (Four pre-existing
failures in `tests/test_harnesses/distill/test_solver_trace.py` on
`origin/main` are unrelated to this change -- no file they cover was
touched.)
