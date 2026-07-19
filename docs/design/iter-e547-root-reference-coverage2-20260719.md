# E547 — moderate strict-subset exposure

E547 evaluates strict-subset sampling multiplier 2, the smallest nontrivial
exposure increase after E546 found multiplier 5 too aggressive. The clean
24-step CPU HF-context continuation starts from E544, sees 15 negative-target
rows (control 7; multiplier 5: 22), processes 1,304 target tokens in 36.48
seconds under `max_wall_minutes=3`, and writes checkpoint SHA
`37002bfd3c63d1ac58f5fc505bf034805b57eee2415d9e15ec1acbb81620fc57`.
It is an explicit no-sync scratch diagnostic.

Late-window positive recall improves from the multiplier-1 control's 0.7014 to
0.7500 and negative accuracy from 0.3333 to 0.4062, while exact-set accuracy
falls from 0.2917 to 0.2083.

On the same OOD `n=4` recipe, multiplier 2 reaches structure 0.2248 and AST
node F1 0.3270, the best values in the multiplier 1/2/5 ladder. Component
recall is 0.2083, matching control and avoiding multiplier 5's 0.0625 collapse.
Fidelity falls to 0.2583, below control 0.4250 and multiplier 5 0.6083. Reward
is 0.5403. The root arity and identity heads each apply six times and change
two choices. Syntax remains 1.0, but meaningful-v1, strict-v2, and AST edge F1
remain 0.0; AgentV fails 0/1 without execution errors.

**Verdict:** use multiplier 2 as the preferred diagnostic exposure setting,
but reject this checkpoint for promotion. More exposure is not the next lever;
the next bounded experiment should recover semantic-role fidelity while
preserving the topology gain. Machine-readable evidence:
[JSON](iter-e547-root-reference-coverage2-20260719.json).
