# Continuous model and harness improvement

Bare `/autotrain` is a non-terminating objective implemented as a sequence of
bounded `ExperimentCampaignV1` cycles. Do not run an unbounded shell process.
Every command still obeys `MAX_RUN_MINUTES`; persistence belongs to the host goal
and the append-only campaign event chains.

## Start or resume

1. Work on a dedicated local branch/worktree. Default authority is local-only:
   no push, PR, merge, paid compute, remote job, or Hugging Face write.
2. Fetch `origin/main` before every decision-bearing run, integrate it safely,
   and record the exact source commit in the locked campaign manifest.
3. Create or resume an unbudgeted persistent host goal when the host supports
   goals. Do not mark it complete because a cycle, queue, or gate finished.
   Report blocked only after the host's repeated-blocker rule is satisfied.
   Hosts without persistent goals finish one bounded cycle and explicitly return
   the loop as active with the exact resume command.
4. Initialize each cycle with loop lineage:

   ```bash
   slm autoresearch init --campaign-id <cycle-id> --loop-id <loop-id> \
     --cycle-index <n> [--predecessor-campaign-id <prior-cycle>] \
     --objective "<falsifiable objective>" --primary-metric <metric>
   ```

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
| Data synthesis feedback | `synthesis-feedback` |
| E*/matrix execution | `running-experiment-matrices` |
| Evaluation/readiness interpretation | `honest-ship-eval` |
| Run evidence and closeout | `documenting-experiment-results` |
| Prior-work/knowledge refresh when needed | `autoresearch` |

After each run and before choosing the next one, print:

```bash
slm autoresearch status --loop-id <loop-id> --matrix
```

The matrix is derived from verified campaign event chains and includes source
commit, diagnosis lane, trainable parameters, primary metric, gates, status, and
next action. Fetch and integrate latest `origin/main` again before the next run.

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
