# R30 ordinary-supervisor diagnostic — incomplete before training

[Machine-readable observation](science-lab-pr-head-r30-20260926-results.json).
Campaign `science-lab-r30-dadeb857f97c-acceptance` binds published commit
`dadeb857f97c70657a49aae0cc598f599e5235ed` and tree
`f76924affe93ae45b189cf81564476a8a2a9cfe3`. The immutable release and execution
copies remain under `outputs/autonomy-integration-20260921/`, not an ephemeral
temporary directory. Inputs reuse the preserved corpus and ancestor checkpoint.

## Locked recipe

CPU scratch TwoTower, 65,826 parameters per arm, seed 7301, eight training
records, six new updates per arm in chunks of three. Learning rates are 0.0003
and 0.0006. The planned paired endpoint uses six public `smoke.eval_nll` cases.
This is diagnostic wiring and measurement evidence, never ship qualification.

## Observed interruption and retained work

Preregistration completed with both manifests locked before execution. Two
bounded ordinary-supervisor invocations ended with exit 137. Inspection
operations completed, but neither arm began training. The second invocation
left a typed `driver_invocation_yielded` result and a retained command cursor;
it did not produce a terminal experiment, evaluation, paired verdict, or new
checkpoint. No AgentEvals or AgentV evaluation bundle exists because evaluation
never began. Interrupted invocations are not passing evidence.

The launcher initially repeated five full source traversals before entering the
supervisor. Its resume path now leaves live source validation to the canonical
supervisor while retaining the connector, preregistration, and input checks.
The second invocation still exhausted its outer cap, so the canonical startup
and finalization path requires further diagnosis. The old cursor and finite
grant are retained; this report does not authorize a reset or a promotion.

The separately retained source-repair verifier exposed hardlinked files in the
host Python runtime. Atomic replacement detached 244 files while preserving
their bytes and modes; the remaining shared links became single links as a
result. The controller recorded an authenticated successor for the changed
runtime identity, retaining its proposal and remaining grant. Runtime changes
must not be silently relabeled as evidence from the original scientific
environment. No source-repair acceptance or operation continuation is claimed.

## Startup profile

A completed read-only instrumented profile measured source identity at 28.21
seconds, Git provenance at 56.80 seconds, and environment identity at 14.32
seconds. Each source/provenance validation reread both source trees: 19,430
regular files, approximately 786 MB. Git provenance validates source again.
The source scan spent 13.58 seconds constructing relative paths in the first
measurement. The profile used the inherited environment rather than the exact
launcher environment. A separate unprofiled repeat exceeded the cap and is not
a completed measurement. Contention is a possible explanation for timing
variance, not an established cause.

[The committed profile](science-lab-r30-startup-profile-20260926.json) preserves
function timings, source, recipe, and limitations. The corrective approach is
to remove redundant path work and combine adjacent source/provenance checks
within one validation boundary, while retaining fresh checks across execution
boundaries and every content, link, mode, and unexpected-file check.

The R31 candidate reduces locked startup from three complete scan pairs to one,
without caching identity across worker boundaries. A retained bounded comparison
observed adjacent source/provenance calls at 11.83 seconds and the combined
validated call at 1.35 seconds, returning identical identities. Sequential cache
and contention differences prevent treating that timing ratio as a confirmed
speedup. [The comparison record](science-lab-r31-source-check-profile-20260926.json)
identifies the unpublished implementation and preserved R30 target. Regression
coverage counts startup scans and changes source bytes with the same length and
restored mtime before the worker boundary; fresh validation must reject the drift.
