# E230 — diverse judged generation roots

Status: **immutable training corpus published; matched train completed; four
ship gates fail; checkpoint rejected**.

E228 improved semantic topology, while E229 showed that another 32 steps on the
same 480-row corpus did not improve the four remaining ship gates. A corpus audit
found that 450 rows were repair/edit derivatives from 91 roots. Although the
mixture configured `rico_real`, no RICO rows were present; `Stack` appeared 574
times and `TextContent` 486 times.

E230 corrects source coverage rather than adding another output-specific loss.
The deterministic build reads the repository's fixture, RICO-train, Awwwards,
ProgramSpec, language-contract, deconstruct, and renderer producers. It disables
repair/edit cloning and caps semantic clusters at two. Every candidate still
passes the independent prompt/output judge before admission. Provenance remains
in record metadata instead of being appended to RICO user prompts, and the
task-aware sampler now treats producer roots without an explicit task as the
output contract's default generation task instead of silently excluding them.

The immutable Git snapshot
`src/slm_training/resources/data/train/e230_diverse_judged_roots_v2` contains 126
records from 192 candidates. Fifteen verifier failures, two quality failures,
four held-out structure overlaps, and 22 excess semantic-cluster members were
rejected. All 126 admitted pairs pass the independent judge and are reachable by
the task-aware sampler. The corpus has 115 semantic clusters (p95 and maximum
exposure 2), 108 structural families, 196 placeholders, and mean quality 0.9937.
Its families are 32 RICO, 17 human-curated, 62 language-contract, 13 ProgramSpec,
one renderer, and one web-distilled record.

The records SHA-256 is
`6c4796159ac0dfccde327586f86b142ac0ff451170a7f757992754b32a65e29d`;
the content fingerprint is
`9f72d85b6cc7118e0f69e010d0debdd2b40ede514e03178dded8e164daaae9bb`.
Synthesis trace `d722ac10571ce461019c4de8b9732a1f` retains the per-record telemetry.

Immediately before training, `origin/main` was fetched. It had zero commits ahead
of the isolated E230 branch, which was clean and one scoped data commit ahead.
The matched train used CPU, 32 steps, batch 4, learning rate 0.0003, seed 0,
frozen local SmolLM2-135M context, lexer output, schema and slot context, no
DESIGN context, compiler-candidate CE plus margin 1.0, and no checkpoint sync.
Capacity-aware no-replacement sampling was used because the generation-root
corpus contains only one edit record; task quotas would otherwise allocate half
the usable task draw to that singleton. The run consumed 18,490 prompt and 7,052
target tokens in 190.60 s. Its 128 draws covered 81 unique rows, including 30
RICO and 25 human-curated exposures. Final loss was 19.1868; bound-component
alignment loss was 2.8894 and margin violation rate 0.5581. Training trace:
`f3e8c41f3dd49e36a3205be941724021`.

Strict compiler-tree evaluation used the unchanged honest slot contract, no
unconstrained fallback, all five suites, AgentEvals JSONL, and AgentV. Two prior
publication attempts were incomplete worktree dependency-path failures (first
AgentV root resolution, then the OpenUI bridge dependency path); the final run
completed with zero AgentV execution errors.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | contract precision | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4642 | 0.2500 | 0.5278 | 1.0000 | 0.8073 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3369 | 0.1567 | 0.2800 | 1.0000 | 0.7330 |
| adversarial | 4 | 1.0000 | 0.2500 | 0.3477 | 0.2083 | 0.2083 | 0.5000 | 0.3870 |
| ood | 4 | 1.0000 | 0.0000 | 0.3750 | 0.2083 | 0.2583 | 1.0000 | 0.7265 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 1.0000 | 0.6865 |

Smoke, held-out, OOD, and RICO exactly match E228; adversarial quality regresses.
The same four gates fail and AgentV remains 1/5 (mean score 0.6). Diverse judged
roots are retained because they repair future data publication and exposure, but
the matched checkpoint is not promotable. The next train must supervise a
request-level component inventory or other schema/AST-derived semantic target;
more duration or another source-mixture adjustment is not supported.

Machine-readable evidence:
[iter-e230-diverse-judged-roots-20260716.json](iter-e230-diverse-judged-roots-20260716.json).
