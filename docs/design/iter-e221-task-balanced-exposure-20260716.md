# E221 — task-balanced exposure launch preflight

Status: **launch failures only; zero training steps; no checkpoint**.

E219 drew 128 examples from the corrected 480-record E218 corpus but exposed
only 29.90 effective records. E221 tests whether the committed task-balanced
mixture reduces that concentration under the otherwise matched E219 recipe.
A five-candidate autoresearch matrix selected the canonical task-balanced policy.

Three preflight failures occurred before model training:

1. The first campaign froze its allowlist before diagnostic checkpoint-policy
   knobs were added, so matrix validation rejected the new knobs.
2. The second campaign compiled bare `python`, unavailable in this environment.
   The shared compiler now uses the active interpreter and persists process-launch
   failures as typed outcomes.
3. The third campaign reached `train_model` but the mixture loader rejected the
   pipeline's canonical `{manifest, diagnostics}` envelope. The loader now accepts
   both that envelope and historical bare manifests.

These are negative harness results, not training results. No optimizer step ran,
no evaluation ran, no checkpoint was written or synced, and no ship claim is
made. The retry must use a fresh immutable campaign because the failed campaign's
single-experiment budget was consumed.

