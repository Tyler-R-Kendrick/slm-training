# E297 — judged semantic-contract data (2026-07-17)

## Hypothesis and implementation

The inherited corpus contained generation prompts that did not determine their
paired outputs. E297 derives a canonical semantic contract from each output AST
(component counts, declarations, reference graph, and placeholders), renders
that contract into the prompt, and makes the independent judge reconstruct and
compare the same contract before admitting a pair.

The immutable published corpus
`e297_semantic_contract_judge_v1` contains 480 records. Ninety under-specified
generation prompts were remediated without changing their outputs; every record
has `independent_judge_passed=true`. Its content fingerprint is
`1e1815e2b658c618b0b556853f7c7b009c384ab98749b3786eebd379a54290a9`.
The records, governance files, stats, and synthesis telemetry are committed
under `src/slm_training/resources/data/train/`, so future runs and the training
data API discover the same version without a local output store.

Publication now copies synthesis telemetry instead of moving it out of the
dataset and rewrites all manifest paths to the source-controlled destination.

Post-run corpus inspection found that the inherited snapshot still contained
29 explicit generation rows and 15 default-generation fixtures that passed the
older prose judge without an exact AST contract. E297 remains immutable as the
historical training input, but future admission now fails closed when any
effective generation row lacks `semantic_contract`; normalization remediates
all such rows, not only edit-derived generation rows. The next corpus version
must therefore rebuild and judge all 119 generation rows before training. A
full 480-record dry normalization confirmed 119/119 generation contracts and
480/480 independent-judge passes under the strengthened gate.

## Bounded recipe

The matched B3 CPU choice arm used width 64, depth 2, seed 0, a 5,000-target
token budget, 200-step ceiling, batch size 2, and configurable
`max_wall_minutes=5`. The branch was clean and zero commits behind
`origin/main` immediately before training. It stopped on the token budget at
107 steps / 5,022 target tokens after 20.44 seconds.

Best weighted NLL was 6.50945, but the loss suite remained incomplete for the
binding category. The scratch checkpoint SHA is
`f74a3c46dc9b5c4cd6c62dcb32aef347f0721fb1e94839cef55e9e4ebf120b26`.
It was not synced or promoted.

## Decoder diagnosis and generalized repair

The train-time evaluation and standalone r1 returned parse 0 despite zero
constrained dead ends. Token traces showed two deterministic-policy defects:

1. grammar-derived object-key names were also admitted as free expressions;
2. JSON-schema `string` acceptance ignored the DSL pack's stronger
   `CONTENT_PROPS` placeholder policy.

E297 removes names from the expression partition while retaining them for
object keys. Positional component contracts are still generated from
`prop_order` and the official JSON schema, but content fields are now annotated
from `CONTENT_PROPS`. Only symbolic slot expressions satisfy those fields.
When no honest slot contract exists, a component whose required content field
cannot be completed is excluded by the exact completion oracle. No component
or property special case was added.

## Results

All three standalone evaluations used the same checkpoint and emitted
AgentEvals JSONL plus pinned AgentV bundles. r1 measured the inherited defect;
r2 isolated the remaining placeholder-policy mismatch after removing free
names; r3 measured the complete generalized repair.

| stage | smoke | held | adversarial | OOD | RICO | dead ends | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train / r1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0/5 |
| r2, name expressions removed | 0.0 | 0.0 | 0.25 | 0.0 | 0.0 | 0 | 0/5 |
| r3, full DSL placeholder policy | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0/5 |

Final r3 quality remains poor:

| suite | n | parse | meaningful | fidelity | structural | reward | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.0 | 0.2519 | 0.0 | 690 / 5,029 ms |
| held_out | 5 | 1.0 | 0.0 | 0.0 | 0.2021 | 0.0 | 210 / 2,703 ms |
| adversarial | 4 | 1.0 | 0.0 | 0.0 | 0.2364 | 0.0 | 194 / 2,754 ms |
| ood | 4 | 1.0 | 0.0 | 0.0 | 0.1906 | 0.0 | 284 / 2,613 ms |
| rico_held | 3 | 1.0 | 0.0 | 0.0 | 0.0727 | 0.0 | 313 / 2,687 ms |

The outputs are grammar-valid but trivial eight-symbol layouts. The r3
latencies are not comparable to parse0 r1/r2 because valid decoding runs 43
model forwards and emits 55 choice tokens per record.

## Verdict and feedback

The deterministic layer now owns the placeholder-valued lexical branch and
restores structural adherence without retraining. That is a harness correction,
not evidence that E297 learned the task.

The judged semantic-contract corpus improves best weighted NLL from E291's
7.09848 to 6.50945 at the matched token budget, but zero meaningful parse,
placeholder fidelity, reward, and AgentV show that lower teacher-forced NLL did
not transfer to semantic generation. E297 is not promotable or ship-ready.

The next quality iteration must first publish the fully remediated 119-row
generation subset under a new immutable corpus version. It should then measure
decision-kind coverage and loss for root component, child occupancy, component
inventory, and slot selection before adjusting the mixture or objective. More
syntax patches or a larger training budget would not address this failure.

Machine-readable evidence:
[`iter-e297-semantic-contract-data-20260717.json`](iter-e297-semantic-contract-data-20260717.json).
