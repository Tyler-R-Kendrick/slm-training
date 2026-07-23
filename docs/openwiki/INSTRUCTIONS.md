A code wiki for the **slm-training** OpenUI layout-SLM research repo.

Prioritize:

1. Concise quickstart for agents and engineers (where to start, what is fixture vs ship).
2. Architecture: TwoTower / grammar-diffusion, harnesses (`train_data` / `test_data` / `model_build`), DSL + grammar backends.
3. Core workflows: train → eval → matrix → docs/MODEL_CARD updates; HF checkpoint bucket sync.
4. Honesty gates (`honest-ship-eval`) vs scratch / quality-matrix demos.
5. Repository organization: `docs/repository-organization.md`, canonical skills, `git mv`, and the repo-policy hooks/CI gate.
6. Agent surface: `AGENTS.md`, `.agents/skills/`, token-efficiency stack, OpenWiki itself.
7. Source map: `src/slm_training/`, `scripts/`, `docs/design/`, playground / Vercel entrypoints.
8. Recursive-denoiser objective semantics: final-depth primary versus explicit
   intermediate/all-depth auxiliary modes, checkpoint migration boundaries, and
   the correction-only SLM-279 evidence under `docs/design/`.
9. Recursive-denoiser health diagnostics: opt-in raw per-depth state/update and
   prediction telemetry, historical `as_is` versus fixture-only
   `residual_delta`, one-forward anytime-depth curves, the exact preregistered
   disposition, and the SLM-282 boundary that fixture success is neither a
   contraction proof nor a quality/ship claim.

Ground pages in repository structure and measured results under `docs/design/`. Prefer practical navigation over generic summaries. Prefer linking existing design docs over rewriting them.

Keep generated pages under `docs/openwiki/`; do not rewrite this `INSTRUCTIONS.md` during routine `--update` runs unless the user asks.
