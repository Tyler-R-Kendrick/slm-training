# DSH2-04 frame-fact prompt paraphrases with leak detector + grounding floor (SLM-365)

**Decision:** supported at the contract-fixture evidence level.
`harnesses/train_data/frame_paraphrases.py` generates the plan-declared style
set — `concise`, `descriptive`, `imperative`, `compositional` — of
natural-language prompt surfaces for one verified `SemanticFrameV1`
(DSH2-02, SLM-363) through the DSH2-03 (SLM-364) provider contract, without
leaking target DSL syntax, marker surfaces, answer identity, or provenance
fields into any prompt. All generation in this issue used the offline
deterministic fixture provider; no network, no real LLM.

Machine-readable evidence:
[`iter-slm365-frame-paraphrases-20260725.json`](iter-slm365-frame-paraphrases-20260725.json).

## Facts only, never the answer

Prompts are constructed exclusively from frame facts. The module renders each
required fact to a plain-language *fact phrase* (`render_fact_phrase`):
component types become lowercase split words (`TextContent` → "a text content
component"), placeholder markers are described by role ("placeholder text for
the text field") and never reproduced, closed values/cardinality/order/state
and runtime relations render as declarative clauses. `_structure_payload`
projects the frame to a nested fact-phrase tree (facts grouped to their
nearest node path, children in declared role order); that tree — plus the
style name — is the entire provider payload. `SemanticFrameV1.canonical_source`
is used only to (a) fingerprint the canonical answer a prompt links to and
(b) build the leak detector. Its surface never enters a prompt. Targets
remain pure canonical symbol-only AST surfaces.

## Declared leak detector

`FrameLeakDetectorV1.from_frame(frame)` rejects any prompt containing:

- **Target DSL syntax** — production forms of every element component in the
  answer (`Card(`, `TextContent(`, …) plus their exact-case bare tokens;
- **Raw marker surfaces** — `:ident`/`$ident` codec forms (generic regex)
  and the answer's exact placeholder values (`:ns.body`);
- **Answer IDs/digests** — `sha256(canonical_source)` and the `content_sha`
  frame projection digest;
- **Provenance fields** — `ast_path`, `schema_field`, `canonical_source`,
  `schema_fingerprint`, `frame_fingerprint`, `prompt_digest`,
  `response_digest`, `cache_key`, `category`.

Known detector characteristic (documented, not observed in fixtures): the
generic marker regex also matches `:ident`-shaped tokens inside free
natural-language content literals; prompts here are built from frame facts
only, so this never fires on fixture data.

## Grounding floor + stop rule

`enforce_grounding_floor` accepts a candidate only when it covers **every**
required fact phrase (normalized containment), asserts **no** forbidden fact
phrase (the excluded-value assertion rendered by the same phrase renderer),
and passes the declared leak detector. A style whose output fails any clause
is dropped with a recorded reason (`missing_required_facts`,
`forbidden_fact_present`, `leak_detected`,
`provider_response_invalid`) — published as a rejected row with detail and
prompt sha, never silently, and never kept by weakening grounding or
tolerating a leak. Tested: a provider whose `imperative` output appends
`Card(` loses the `imperative` style with a recorded `leak_detected` reason
while the other three styles publish.

## Diversity + dedup

Per frame/style/provider the set publishes: token count, distinct-token
count/ratio per style, pairwise token-multiset Jaccard across styles, an
aggregate `lexical_diversity` (1 − mean pairwise Jaccard), and
`structural_diversity` (distinct style-template signatures / rows), plus
accepted/rejected counts per style and rejection counts per reason.

Card fixture (11 required facts, 2 components, 1 placeholder):

| Metric | Value |
| --- | --- |
| accepted per style | 1/1/1/1 (concise/descriptive/imperative/compositional) |
| pairwise token Jaccard | 0.7838 – 0.8824 |
| lexical_diversity | 0.161889 |
| structural_diversity | 1.0 |

The modest lexical diversity is honest and expected: the grounding floor
requires every style to cover the *same* fact phrases verbatim, so content
tokens are shared by construction; diversity comes from template connective
prose and tree structure. Buying more lexical spread by paraphrasing the
fact phrases themselves would weaken exactly the grounding this issue
requires — the stop rule forbids it.

`dedupe_paraphrase_rows` drops, with reasons: exact duplicates (any style
pair), near-duplicates (same-style normalized-token Jaccard ≥ 0.9), and
repeated style-template outputs (same-style template signature = style +
sorted coverage keys). Near/template rules are same-style only, so genuinely
distinct styles of one frame are never deduplicated against each other.
`generate_answer_family_paraphrases` runs the same dedup across frames: two
surface-different but equivalent sources of the card fixture yield 4 kept
rows + 4 counted `exact_duplicate` rejections, all linked to one canonical
answer fingerprint.

## Linking

Every accepted row carries `canonical_answer_fingerprint`
(`sha256(canonical_source)`) on its set plus the accepted equivalence-set
fingerprints — fingerprints, never answer text. Each row's
`style_provider_provenance` embeds the DSH2-03 contract record
(`ParaphraseRequestV1`/`ParaphraseResponseV1`: provider/model/revision,
schema + frame fingerprints, system/prompt/response digests, sampling, seed)
and every raw response passes `validate_raw_response` + secret redaction
before grounding checks.

## Evidence

| Control | Result |
| --- | --- |
| all four plan-declared styles generate from a frame fixture | pass |
| every accepted row covers all required facts | pass |
| style missing a required fact rejected with reason | pass |
| forbidden-fact assertion rejected | pass |
| leak detector catches DSL syntax / marker / answer-id / provenance leaks | pass |
| stop rule drops a leaking style with recorded reason | pass |
| targets stay symbol-only (answer surface never in prompts) | pass |
| diversity metrics published per frame/style | pass |
| dedup exact + near-duplicate + repeated template | pass |
| distinct styles of one frame never dedup-collapsed | pass |
| cross-family dedup links one canonical answer | pass |
| rows link canonical answer + equivalence fingerprints | pass |
| per-row provenance uses the DSH2-03 contract | pass |
| determinism via the offline provider (fixed clock/seed) | pass |

Focused new-module suite: **29 passed**
(`tests/test_harnesses/train_data/test_frame_paraphrases.py`). Ruff check and
`ruff format --check` pass on both new files,
`scripts.verify_version_stamps --check` is ok (the new
`harness.experiments.slm365_frame_paraphrases` component v1 covers all four
new paths; DSH2-02/03 component entries were left to their owners), and
`git diff --check` passes. `scripts.repo_policy` reports one failure with a
foreign cause: an untracked top-level `requirements.txt` left in the shared
worktree by another agent (present before this change began; not an SLM-365
path; deliberately left in place rather than deleting another agent's file).
No SLM-365 path was flagged. Combined suite results:
`tests/test_harnesses/train_data` 121 passed / 4 failed (all four in
`test_staged_materialization.py`); the DSH2-relevant dsl subset
(`test_canonicalize`, `test_parser`, `test_semantic_frame`,
`test_paraphrase_provenance`) 57 passed / 2 failed (both in
`test_semantic_frame.py`). All six failures are **pre-existing foreign
failures**: they reproduce identically with the two SLM-365 files removed
from the tree (verified), and none of their files import or are imported by
this change. The full `tests/test_dsl` directory exceeded the 170s
per-command cap, so the focused DSH2 subset above was run instead.

## Honest limitations

- No real LLM API was called (no network, no credentials, no cost); all
  generation used the offline `FrameParaphraseFixtureProviderV1`, which
  renders deterministic style templates over the module-rendered fact
  phrases. A hosted provider can be substituted through the same DSH2-03
  protocol without changing the detector, grounding floor, dedup, or
  linking machinery — its outputs simply face the same acceptance bar.
- Fixture scale only: two small openui fixture frames (Card/TextContent,
  Callout). No corpus build, train, eval, benchmark, checkpoint, AgentEvals
  publication, or ship claim was produced.
- Fact-phrase rendering is declarative-template English, not fluent copy;
  fluency is the (out-of-scope) job of a real paraphrase provider, and the
  coverage check is deliberately phrase-containment based so any provider's
  output is judged against the same required phrases.
- Inter-style lexical diversity is structurally bounded by the shared
  required-fact phrases (see "Diversity + dedup"); the metrics report this
  honestly rather than inflating it.

## Research lineage

Continues the DSH2 chain: DSH2-02 (`dsl/semantic_frame.py`) supplies the
verified fact partition this module renders; DSH2-03
(`dsl/paraphrase_provenance.py`) supplies the provider protocol, provenance
record types, response validation, and redaction reused verbatim here. The
content-addressed fingerprints reuse
`harness_core/lineage/records.content_sha`/`canonical_json`, and the
per-record diversity metrics complement (not duplicate) the corpus-level
fingerprints of `harnesses/train_data/diversity.py`, which operate on
materialized `ExampleRecord` rows downstream of any future synthesis that
might consume these prompts.
