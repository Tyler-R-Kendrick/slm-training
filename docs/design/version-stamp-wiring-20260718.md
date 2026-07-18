# Version-stamp wiring validation

**Date:** 2026-07-18
**Status:** wiring harness / fixture-only evidence. No training run, no new
checkpoint, no quality or ship claim.
**JSON:** [version-stamp-wiring-20260718.json](version-stamp-wiring-20260718.json)
**Contract:** [version-stamp-contract.md](version-stamp-contract.md)

## Summary

- Introduced the normalized component-version registry
  (`src/slm_training/resources/versions.json`, `version_registry/v1`, 15
  components) and the `version_stamp/v1` envelope emitted by every canonical
  result writer via `slm_training.versioning.build_version_stamp`.
- End-to-end probe: `evaluate_model --suite smoke` on the committed
  `playground_demo` fixture checkpoint against the `ci` eval snapshot
  (`--eval-limit 4 --gen-steps 4 --max-attempts 1`, CPU) wrote
  `outputs/runs/stamp_probe/eval_smoke.json` + `eval.json`, each carrying the
  stamp with the real code commit and
  `{harness.model_build.eval: v1, evals.meaningful_program: 2.0.0, evals.scoring: v1}`.
  Probe numbers (n=3 diagnostic subset, parse 0.33) are wiring evidence only.
- Enforcement probes: an unbumped `ship_gates.py` edit was blocked by
  `verify_version_stamps --check` naming `gates.ship` with both remedies; a
  same-version `no-bump:` history note un-blocked it; the PostToolUse hook
  nudged on the edit and went silent once the registry entry was touched.
- Staleness/grandfathering: `--stale` reports 0 stale, 118 legacy unstamped
  `docs/design` files (reported, never errors), and the probe's stamped
  outputs as fresh under `--include-outputs`.

## Honesty caveats

- Fixture checkpoint + diagnostic subset: nothing here is a model-quality
  measurement; the record exists to validate the versioning contract wiring
  and to be the first stamped entry in the ledger.
- Legacy `docs/design` JSONs remain unstamped by design (grandfathered);
  they surface in `--stale` as `legacy_unstamped`, not as retest candidates.

## Next steps

- On the next real bump of any component, run
  `python -m scripts.verify_version_stamps --stale --component <id>` and
  either re-run the listed experiments or label their matrix rows invalidated.
- Follow-ups deliberately out of scope: threading stamps through the
  checkpoint-reference `metadata` tuple, dashboard surfacing, `run_id`
  normalization.
