# E607 — verified semantic-plan root score trace

Date: 2026-07-20
Status: completed diagnostic; not promotable or ship

E607 adds bounded score decomposition for every verifier-approved semantic-plan
root token. It does not change decode scores. The capped OOD `n=4` replay uses
the exact E606 margin-2 policy and checkpoint.

The run completes normally and reproduces E606 exactly: syntax 1.0,
meaningful-v1 0.5, strict meaning-v2 0, fidelity 0.5917, validity 0.7550,
structure 0.5756, component recall 0.6250, reward 0.8145, AST-node F1 0.6111,
AST-edge F1 0.4643, and AgentV 0/1.

The trace localizes the remaining reachability failure:

- after dashboard emits Button, Callout, Card, and Card, the verifier approves
  `+Stack`; its base score is −44.47 and the configured +8 leaves it −36.47,
  still 65.35 points behind `TextContent`;
- gallery receives a verifier-approved `+Stack` at 15 decisions, but the
  post-bias deficit remains 79.32–87.03 points;
- modal's competitive `+Stack` wins and its complete planned closure reaches
  EOS;
- auth's first Stack closure succeeds, but one planned EOS remains below a
  placeholder, after which the verifier constructs and terminates a duplicate
  Stack.

This rejects verifier abstention and downstream overwrite hypotheses. The
fixed +8 root score has the same family-specific scale problem E605 found for
component selection. The next bounded experiment should floor each
verifier-approved root token above the best legal score, including EOS, while
retaining the existing exact completion validation.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e607-semantic-plan-root-trace-20260720.json).
