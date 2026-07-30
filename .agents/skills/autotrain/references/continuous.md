# Continuous model and harness improvement

Bare `/autotrain` is a non-terminating objective implemented as a sequence of
bounded `ExperimentCampaignV1` cycles. Do not run an unbounded shell process.
Every command still obeys `MAX_RUN_MINUTES`; persistence belongs to the host goal
and the append-only campaign event chains.

## Start or resume

1. Work on a dedicated local branch/worktree. Default authority is local-only:
   no push, PR, merge, paid compute, remote job, or Hugging Face write.
2. Close and commit the preceding cycle, then fetch and merge before every
   decision-bearing run:

   ```bash
   git fetch origin main
   git merge --no-edit origin/main
   git status --short
   git rev-parse origin/main
   git rev-parse HEAD
   ```

   Resolve semantic conflicts by preserving both upstream and cycle work. Repair
   every failing repository check, including a failure reproduced on the fetched
   baseline, before training resumes. Never rebase away experiment provenance or
   begin a run with unresolved conflicts or tracked dirt.
3. Create or resume an unbudgeted persistent host goal when the host supports
   goals. Do not mark it complete because a cycle, queue, or gate finished.
   Report blocked only after the host's repeated-blocker rule is satisfied.
   Hosts without persistent goals finish one bounded cycle and explicitly return
   the loop as active with the exact resume command.
4. Initialize each cycle with loop lineage:

   ```bash
   slm autoresearch init --campaign-id <cycle-id> --loop-id <loop-id> \
     --cycle-index <n> [--predecessor-campaign-id <prior-cycle>] \
     --upstream-commit <fetched-origin-main-sha> \
     --integration-commit <merged-head-sha> \
     --objective "<falsifiable objective>" --primary-metric <metric>
   ```

   Initialization verifies that `upstream_commit` is the current fetched
   `origin/main`, `integration_commit` is the clean checked-out `HEAD`, and the
   latter contains the former. The locked `ExperimentCampaignV1.source_commit`
   must equal that integration commit and must declare `source_dirty=false`.

## Iterate forever

Run one bounded campaign through research, hypothesis, preregistration, execution,
diagnosis, documentation, and local promotion. After every result, audit both the
model and the harnesses. A harness lane opens only when a typed `HarnessSignalV1`
was reproduced on the frozen input and names exactly one canonical family:
`autoresearch`, `annotations`, `distill`, `experiments`, `model_build`,
`preference`, `quality`, `rl`, `test_data`, or `train_data`.

Route that lane through `improve-openui-harnesses`, repair the canonical owner,
then replay the identical model/data arm. Never combine a harness change and model
change in the same attribution arm. Unreproduced suspicions remain evidence; they
do not authorize harness edits.

Coordinate the existing owners instead of reimplementing them:

| Signal or stage | Required owner skill |
| --- | --- |
| Campaign execution and typed feedback | `openui-autoresearch` |
| Canonical harness diagnosis/repair | `improve-openui-harnesses` |
| Lean metric certificate replay, band miss, or proof/assumption repair | `improve-lean-optimums` |
| Data synthesis feedback | `synthesis-feedback` |
| E*/matrix execution | `running-experiment-matrices` |
| Evaluation/readiness interpretation | `honest-ship-eval` |
| Run evidence and closeout | `documenting-experiment-results` |
| Prior-work/knowledge refresh when needed | `autoresearch` |

When a campaign carries `metric_expectations_sha256`, replay its v2 certificate
through the in-repo LeverProof checker before promotion or successor selection.
An in-band result continues. An assumption-backed miss blocks promotion and
requires `measurement_control`, `training_method`, `architecture`, `lean_model`,
and `assumptions` in both the candidate matrix and ranked priorities. A
theorem-backed contradiction stops that campaign and moves the still-active outer
goal into measurement/formal-model repair; ordinary training does not resume until
the contradiction is repaired. Historical v1 certificates are display-only.

After each run and upstream-integration repair, and before choosing the next run,
print the default five-cycle view:

```bash
slm autoresearch status --loop-id <loop-id> --matrix --last 5
```

It renders three tables derived from verified campaign event chains: run results,
diagnostic/harness/Lean signals, and evidence-linked speculative next-run
priorities. Use `--all` for complete history. Fetch and merge latest `origin/main`
again before the next bounded run.

## Automated promotion and judge changes

No human approval is required. Ordinary researcher/hypothesizer or harness changes
promote locally only after their frozen benchmark and held-out gates pass.
Changing a metric, threshold, gate, evaluator, or frozen case requires a separate
preregistered `ExperimentCampaignV1` meta-campaign with unchanged held-out control
cases. It may add a stricter case or repair a demonstrably invalid judge; it may
not lower/delete a gate, train on frozen cases, weaken decode invariants, or modify
the meta-gate evaluating itself. A rejected meta-campaign leaves the judge
unchanged and the loop selects another candidate.

Every executed run still follows `documenting-experiment-results`; data builds
also follow `synthesis-feedback`, matrices follow `running-experiment-matrices`,
and readiness claims follow `honest-ship-eval`.
