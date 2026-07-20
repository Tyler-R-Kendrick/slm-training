---
type: experiment-idea
status: staged        # staged | filed | running | resolved
tags: [idea]
created: {{date}}
primary_metric:       # the falsifiable metric this would move
knobs:                # campaign-allowed ExperimentKnobs fields only
sources:              # [[source-note]] wikilinks backing the idea
linear:               # SLM-N once filed
---

# {{title}}

## Hypothesis

If we change <knob> then <primary_metric> improves because <mechanism>.
State it so a single run can falsify it.

## Grounding

- Prior work: `[[source-note]]` / lineage row
- Prior evidence in-repo: `docs/design/*` rows (matched controls, if any)
- Novelty check: not a finished knob signature and not a recorded dead-end

## Plan

- Matrix members (≥5 when this becomes a campaign) and the recommended one
- Controls to hold fixed
- Expected artifact root: `outputs/autoresearch/<campaign>/`

## Filing

When staged → filed, create the Linear issue (team `SLM`) and, on run, the
`docs/design/iter-*.md` record. Link both here and flip `status`.
