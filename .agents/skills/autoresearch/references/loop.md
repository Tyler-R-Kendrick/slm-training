# The autoresearch loop (orchestration)

This is the coordinating stage. It decides *what to do next* and delegates every
concrete action to an owner: brains, OpenWiki, discovery, the autotrain
hypothesis loop, and Linear. Read [contracts.md](contracts.md) alongside it.

## Inputs

- A research objective, or "advance the program" (then pick the highest-value
  open question from [`docs/brains/repo/MOC.md`](../../../../docs/brains/repo/MOC.md)).
- Current state: active Linear projects/issues on team `SLM`, recent
  `docs/design/iter-*.md` rows, and the brain's open-questions / dead-ends lists.

## The loop

1. **Orient from memory.** Read the repo brain MOC and any personal-brain notes
   for the objective ([brains.md](brains.md)). If OpenWiki is stale for the
   surfaces involved, refresh it ([openwiki.md](openwiki.md)). Capture the
   current open questions and known dead-ends — they bound everything downstream.

2. **Discover prior work.** Run the discovery stage ([discovery.md](discovery.md))
   to gather related research and in-repo lineage for the objective. Write/refresh
   `source` notes in the brain; do not duplicate `research-lineage.md` — link it.

3. **Synthesize → open questions.** Update the brain: which questions are now
   answered by measured `docs/design/` evidence, which are newly opened by the
   discovery pass, and what the sharpest falsifiable next question is.

4. **Hypothesize.** Feed the open questions + grounded sources into the autotrain
   hypothesis loop ([hypothesis.md](hypothesis.md)). It returns a novelty-audited,
   grounded matrix (≥5 candidates, one recommended). Reject any candidate that
   matches a recorded dead-end or a finished knob signature.

5. **File as Linear work.** Turn the surviving hypotheses into tracked work
   ([linear.md](linear.md)): a new **project/initiative** for a research thesis,
   **milestones** for tracks, and one **issue** per experiment. Link each issue
   to its brain `experiment-idea` note and (on run) its `docs/design/iter-*.md`.

6. **Hand off execution.** Running experiments is `autotrain` +
   `openui-autoresearch`, not this skill. Do not launch paid GPU / remote jobs or
   HF writes without explicit user approval.

7. **Fold results back.** A run counts as landed only once its evidence exists per
   the pipeline's iron law: a committed `docs/design/` record (JSON + markdown)
   with recipe metadata and honest pass/fail vs gates (`documenting-experiment-results`;
   readiness language → `honest-ship-eval`). From that record, update the brain
   (answered questions, new dead-ends, lineage graduation), flip the Linear issue
   and `experiment-idea` note status, and run `synthesis-feedback` after any data
   build. Then return to step 1.

## Where each thing lives (never cross the streams)

| Kind of fact | Home | Owner |
| --- | --- | --- |
| Idea / synthesis / open question / thesis | `docs/brains/` | this skill |
| Codebase navigation | `docs/openwiki/` | `openwiki` CLI |
| Measured result | `docs/design/` | `documenting-experiment-results` |
| Cited-paper → code lineage | `research-lineage.md` + `resources/autoresearch/*.json` | discovery / lineage |
| Tracked work item | Linear (team `SLM`) | `linear.md` |
| Campaign evidence bundle | `outputs/autoresearch/<campaign>/` | `autotrain` / `openui-autoresearch` |

## Worked example (one turn of the loop)

Objective: "reduce evaluation cost without regressing the five-suite scoreboard."

1. **Orient** — repo brain MOC lists an open question "can we halve eval cost?"
   and a dead-end (a caching lever that regressed a gate). OpenWiki eval page is
   current, so no refresh.
2. **Discover** — `research-lineage.md` + a web sweep surface an eval-caching
   paper; write a `source` note (Adapted, "take the cache-key idea, not the
   serving scheduler").
3. **Synthesize** — sharpen the open question to a falsifiable one: "does keying
   the eval cache on `<X>` cut wall-clock ≥40% with zero gate regression?"
4. **Hypothesize** — stage an `experiment-idea` note, feed it + the source into
   `slm autoresearch hypothesize`; it returns a validated 5-candidate matrix.
5. **File** — under the existing eval program: one Linear `SLM-N` issue for the
   recommended candidate, linked to the `experiment-idea` note; excess candidates
   stay as brain notes, not issues.
6. **Hand off** — hand the campaign to `autotrain`; do not run paid jobs here.
7. **Fold back** — when `docs/design/iter-*.md` lands, mark the question answered
   or add a dead-end, and flip the issue + note status.

## Stop conditions

- No hypothesis survives the novelty audit → record why in the brain and stop;
  do not file low-signal issues to look busy.
- The objective needs a run → hand off; this loop's deliverable is *tracked,
  grounded work + updated knowledge*, not the run itself.
