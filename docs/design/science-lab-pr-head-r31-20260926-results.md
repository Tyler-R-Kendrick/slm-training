# R31 ordinary-supervisor diagnostic: incomplete bounded invocations

The first bounded invocation on published commit `43327120eda92ef9aff2a7eb89d0777e0849317b` ended with exit 137. Inspection succeeded in 36.2266 seconds and the driver operation started. By the observation cutoff, no training arm, evaluation arm, new checkpoint, paired endpoint, or durable driver command cursor had completed. Runtime heartbeats are not measurement evidence.

The [machine-readable observation](science-lab-pr-head-r31-20260926-results.json) binds the immutable source, locked preregistration, retained logs, and recipe. The CPU scratch diagnostic plans six updates per arm, chunks of three, seed 7301, eight training records, 65,826 parameters per arm, learning rates 0.0003 and 0.0006, and six public smoke evaluation cases per arm. No ship gate, promotion, or capability improvement is established. No evaluation result exists to publish through AgentV for this invocation.

A successor approach preserves the same campaign, grant, and source while serializing the retained campaign launcher and background verifier with one host-compute lock. Observed host memory and I/O pressure motivates this operational change; it does not prove contention caused the interruption. Busy admission returns before starting the supervisor. The first-invocation observations above remain unchanged; the second invocation is recorded below. No checkpoint was created, so no new model-card roster entry is warranted.

## Second bounded invocation

After the preceding verifier reached an authoritative terminal state, the campaign acquired the shared host-compute lock. The verifier timer was restored while the campaign retained the lock. This invocation ended with exit 124. The captured traceback stops during the supervisor's initial source validation: `runtime_source_provenance` → `_runtime_manifest` → `_files(release)` → `path.read_bytes`. No new inspection, training, evaluation, checkpoint, or command-cursor result was recorded. Several preceding lock-busy admissions returned 10 before starting the supervisor; they are not training attempts or successful measurements.

Serialization alone did not resolve startup exhaustion. The successor investigation targets bounded-memory content hashing and the actual startup validation path while preserving fresh full-source validation, immutable evidence, the original grant, and the hard run cap. This replaces the contention-only hypothesis; no successful continuation is claimed.

## Host memory remediation

Host inspection found five loaded Pyright language servers for the same repository, each using approximately 3 GB, under duplicate Serena MCP servers owned by the same Codex process. Memory pressure reported full stalls averaging 35.42% over 60 seconds and 58.83% over 300 seconds. This provides a concrete resource-contention mechanism, but does not by itself prove the sole cause of every timeout.

After checking parent PID, process start time, launch mode, and working directory, four older duplicate Serena servers received SIGTERM. All four servers and their Pyright children exited; the newest loaded server remains. No source, credentials, training process, campaign, or continuation grant was changed. Available memory then measured 17,450,200 KiB. The same campaign is resumed to test whether this operational fix changes the outcome.

## Third invocation: control training progressed

After duplicate-server cleanup, the same locked campaign advanced. Inspection completed in 14.7280 seconds. The driver registered its cycle, committed the first training chunk, and recorded the second chunk checkpoint. The control train summary reports six updates, `stopped_on=steps`, eight records, and exact same-environment continuation from step three. Last training loss is 40.56913757324219, not the confirmatory evaluation metric. Two immutable control checkpoint bundles are retained.

The enclosing invocation ended with exit 137. The durable command cursor and runtime `driver_yielded` event establish resumable progress, not end-to-end completion. At this cutoff neither evaluation arm nor candidate training completed. Continuation preserves source, manifest, checkpoints, and remaining grant. No promotion, shipment, or capability gain is claimed.

Control checkpoint SHA-256: `673fce9d06eee0b226d6520c1a8f32c9dacbc924ac8997d15a5e765df2b465b8`. The model card and README record this local scratch checkpoint.

## Fourth invocation: explicit cursor reconciliation prerequisite

The next ordinary-supervisor invocation returned 10 with `driver_attempt_requires_reconciliation` and the `driver_continuation_reconciliation` capability. The original driver operation is waiting rather than reported as completed. Its pending artifact is `a817c549b1189e19c9809d134545d239bcecbc6643c574b5ad512200ed6d62ad`.

Source inspection identifies a recovery mismatch to fix: the locked first evaluation command has `--partial-scoreboard` and `--max-records-this-run`, but no `--resume-run`. Recovery of an interrupted reservation after a committed training prefix requires `--resume-run` already in that original command. The successor must construct and validate the explicit evaluator continuation without repeating training, overwriting partial rows, resetting the grant, or treating an interrupted reservation as scientific success. The retained campaign and checkpoints remain intact pending this repair.

The ordinary runtime recorded a genuine `operation_repair_requested` event at 21:07:18.979078Z for `supervisor-1-driver-c27384e5916102e8`, artifact `8cddf0b0934ffc8a384e0cdff4557981dee156c234297224bd8c880cbc2d3b44`. Thus this failure reaches the durable repair queue. Agent execution, independent acceptance, publication, and original-operation continuation remain unproven; queue admission alone does not discharge them.
