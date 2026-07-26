# autotrain_wf_smoke_20260726_iter146

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v3 last_loss=30.91221809387207 stopped_on=steps wall=61.48 max_wall=2.5833333333333335 n=3

First iteration on `wf_smoke_v3` (fresh fixture build; prior `wf_smoke_v2`
artifacts under `outputs/` do not persist across containers). Carries the
`harness.train_data` v21 canonical-template-marker fix
(`canonicalize_example_template_markers` + `assert_canonical_template_markers`
in `_normalize_record`, cherry-picked from the unmerged canonicalization fix)
so a from-scratch fixture build does not regress into the iter96
`canonical_template_marker_gate` failure. `wall` sums the full
data_train+data_test+sft+eval phases (7.02s + 3.38s + 8.48s + 42.61s); `n=3`
is the smoke suite (diagnostic subset). `meaningful_program_rate=0.0`,
`decode_timeout_count=1` — wiring-only, ship gates not requested.
