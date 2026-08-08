# SGS-005 — Production / oracle / evaluation authority projections

**Issue:** [SLM-443](https://linear.app/quickdeploy-ai/issue/SLM-443)
**Status:** implemented (unit fixtures)

## API

`project_prompt_requirements(reqs, mode=...)` in
`src/slm_training/data/semantic_plan/requirements_project.py`

| Mode | Behavior |
| --- | --- |
| `production` | Rejects `oracle_override`; caps authority via weaker-only combine with `advisory-learned` |
| `oracle_diagnostic` | Allows oracle override; never escalates authority |
| `evaluation_only` | Forces `evaluation-only` (non-promotable) |

`requirements_may_hard_prune` is False unless
`EvidenceKind.COMPILER_AUTHORED_CERTIFIED` is also supplied.
`project_semantic_plan` reuses `SemanticPlanV1.to_production_dict`.

Projections do not alter the exact legal domain.
