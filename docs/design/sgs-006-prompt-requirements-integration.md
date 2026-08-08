# SGS-006 — Prompt requirements integration

**Issue:** [SLM-454](https://linear.app/quickdeploy-ai/issue/SLM-454)
**Status:** implemented (unit fixtures)

## API

`src/slm_training/data/semantic_plan/requirements_integrate.py`

- `score_plan_against_requirements(plan, requirements)` → satisfied / violated / unknown per fact
- `annotate_actions_with_requirements(actions, requirements)` → soft features only (action count unchanged)
- `requirements_hard_prune_allowed` → false unless `COMPILER_AUTHORED_CERTIFIED` evidence is also present

Absent requirements → baseline no-op. No gold/target consultation.
