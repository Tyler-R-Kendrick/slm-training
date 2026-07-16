# E230 — diverse judged generation roots

Status: **immutable training corpus published; model train pending**.

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
No model or quality result is claimed yet.

Machine-readable evidence:
[iter-e230-diverse-judged-roots-20260716.json](iter-e230-diverse-judged-roots-20260716.json).
