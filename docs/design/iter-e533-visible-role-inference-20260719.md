# E533 — honest visible-role inference

E533 audits the train/inference authority gap exposed after E531. E530 training
prompts contained normalized `Components:` and `Semantic roles:` lines, while
the matched E532 OOD prompts contained neither. The new opt-in projection uses
only official component names already mentioned in user prompt prose and the
honest visible slot contract. It never reads gold component inventory or the
gold reference graph and fails closed when either visible source is absent.

The four OOD prompts produce these partial visible inventories:

| Record | Prompt-mentioned components | Typed role evidence |
| --- | --- | --- |
| dashboard | `Callout`, `Card` | status title/body → `Callout` |
| gallery | `ImageGallery` | none |
| modal | `Modal` | title → `Modal` |
| auth | `Button`, `Input` | name/email → `Input` |

E533 holds the E531 checkpoint and every E532 evaluator setting fixed, adding
only `--semantic-role-contract-in-context`. The CPU OOD n=4 diagnostic
completed under the 170-second process cap and emitted AgentEvals JSONL plus an
AgentV SDK bundle.

| Metric | E532 control | E533 visible roles | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate v1 | 0.0000 | 0.0000 | 0.0000 |
| Placeholder fidelity | 0.4667 | 0.3833 | -0.0833 |
| Structural similarity | 0.1431 | 0.1159 | -0.0273 |
| Component type recall | 0.2917 | 0.2292 | -0.0625 |
| Reward | 0.3685 | 0.3685 | 0.0000 |
| AST node F1 | 0.2543 | 0.1627 | -0.0916 |
| AST edge F1 | 0.0455 | 0.0417 | -0.0038 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

Reject the projection as a quality lever and do not spend another bounded
train on it. Keep the opt-in harness because it makes future contract-trained
evaluations honest and distribution-matched, but move the next causal lever
away from additional prompt conditioning. Explicit grammar/reference
construction must improve topology without hidden gold inputs.

Machine-readable evidence is in
[the E533 JSON](iter-e533-visible-role-inference-20260719.json).
