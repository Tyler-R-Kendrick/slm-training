# SLM-588: finite paired summaries cannot manufacture an effect

## Reproducer and fix

Current imported `paired_record_screening` accepted six distinct, declared-root
pairs whose finite deltas were all `1e308` against `minimum_effect=1.5e308`.
The even-sample median and naive sum overflowed to infinity; `win` became true
despite the real median being below the unchanged threshold.

`PairedRecordDeltas` now uses stdlib exact-accumulation `mean` for the mean
and central order-statistic midpoint. An unrepresentable sample SD raises
`ValueError("invalid_evidence: paired dispersion overflow")`; it is not
converted into a negative or completed score. No clipping is introduced.
The public paired consumer, climb measurement, and corrective-data experiment
already call this owner; no new scheduler or evaluation path is added.

Alpha, non-tied-pair floors, effect thresholds, identity matching, independence
requirements, and promotion/ship authority are unchanged. Historical evidence
is not rewritten. Corrected floating summaries can differ from historical
rounding; replay original rows under the successor source identity rather than
pooling old/new summaries as identical evidence.

## Executed evidence

[Machine evidence](slm588-paired-summary-overflow.json) records source hashes and
commands. All six new cases failed before the patch. The local complete
paired-statistics file then passed **37 tests, zero skips, 0.17s**; Ruff check
and format check passed. A runtime mutation restores the old midpoint producer:
the unchanged effect-gate oracle fails with the expected AssertionError, and
the test checks that the mutated producer actually ran once.

These are arithmetic fixtures, not model-quality measurements or a promotion.
No training, external agent, service, hosted workflow, or paid job ran.

Broader runs were not green: an other-owned positional-pairing mutation fixture
failed before mutation; four sample-sizing tests observed unexpected corpus/power
state; a later identity-property run hit an unchanged 200ms Hypothesis deadline
(745.82ms first call versus 3.14ms replay). None is waived or xfailed here.
Full release verification remains outstanding.

## Publication and source qualification

Local execution: dirty candidate at `e0eca9f9910244ecc20eb480d363f1852417b599`.
PR base: `a809e81878302baa27a9d6fedd8f0b79ee3af857`. Only the arithmetic diff is published;
the patched source bytes match those tested locally. The PR preserves the
remote test file and adds the six new cases rather than publishing unrelated
dirty test edits; therefore 37 passing local tests are **not** a claim that
the remote base's entire test file or merge candidate passed.

The fetched base version registry has no watched path for this paired owner;
its existing registry is preserved. The integrator should bind successor
evidence to the corrected source digest and apply its composed ownership/version
mapping. No scientific policy or historical stamp is changed in this fix.
