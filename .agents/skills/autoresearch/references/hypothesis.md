# Hypothesis loop (the auto training hypothesis loop)

Generate falsifiable, grounded, novelty-audited hypotheses by driving the
**autotrain** hypothesis loop. This skill does not reimplement hypothesis
generation — it feeds curated knowledge in and files the survivors out.

## Where it runs (owned elsewhere; this skill drives it)

The hypothesizer is a phase of `autotrain`
([`.agents/skills/autotrain/references/autoresearch.md`](../../autotrain/references/autoresearch.md));
its campaign methodology, provider rules, `init` / `research` / `validate` /
`run` sequence, and evidence contracts are owned by **`openui-autoresearch`**.
This skill drives the one **hypothesize** step and consumes its matrix — it does
not re-document the campaign commands. The two provider paths it invokes:

```bash
slm autoresearch hypothesize --campaign-id <id> --provider agent --matrix <json>  # code-capable agent
slm autoresearch hypothesize --campaign-id <id> --provider openai                 # Responses + Structured Outputs
```

Validation, planning (`run` without `--execute`), and execution are the campaign
owner's steps — hand off to `openui-autoresearch` for them.

## Feed it curated knowledge

Before calling `hypothesize`, assemble from the brain + discovery stage:

- the sharpest **open questions** for the objective (`docs/brains/repo/MOC.md`);
- the grounded **sources / levers** with fidelity labels (from
  [discovery.md](discovery.md));
- the recorded **dead-ends** and finished knob signatures to exclude.

Stage each run-ready idea as an `experiment-idea` note first
(`docs/brains/templates/experiment-idea.md`) — that note is the hypothesis's
brain home and the thing the Linear issue links to.

## Validate before filing

A matrix is acceptable only when it has:

- ≥5 distinct candidates and exactly one recommended member;
- grounded citations and evidence-role coverage;
- categorical novelty audits — **no candidate matches a recorded dead-end**, a
  finished knob signature, or a prior campaign experiment ID;
- only campaign-allowed `ExperimentKnobs` fields (never researcher-authored
  shell/code).

These are the acceptance criteria this skill checks a matrix against before
filing. The `validate` / plan / execute commands that enforce and run them are
`openui-autoresearch`'s; executing (paid GPU / remote / HF writes) needs explicit
user approval.

## Hand-off

- Surviving, validated hypotheses → file as Linear work ([linear.md](linear.md)).
- Actually running them → `autotrain` + `openui-autoresearch`.
- Completed experiments become **typed hypothesizer feedback**; the next matrix
  must link its predecessor and avoid finished signatures. The loop improves via
  evidence and typed feedback only — never by rewriting its own code, frozen
  cases, or acceptance thresholds.
