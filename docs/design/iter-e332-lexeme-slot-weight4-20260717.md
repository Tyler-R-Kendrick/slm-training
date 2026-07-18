# E332 frozen lexical slot weight 4 — 2026-07-17

E332 raises only the frozen E326 slot-component decode weight from 1 to 4.
The intervention is motivated by correct high-margin Callout owner logits that
were not overcoming base decoder scores.

Smoke component recall rises 0.3333→0.50 and structure 0.5464→0.6281.
Limited-RICO structure rises 0.4826→0.6717. OOD meaningful/reward fall from
1.0/0.9857 to 0.75/0.7425 but remain above their honest gates. Parse and
visible-slot fidelity remain 1.0 on every suite.

AgentV passes 5/5 and all current scratch ship gates pass.

**Verdict:** accept decode weight 4 as the scratch serving policy and persist
it in E333. This is not a production claim: the checkpoint is local scratch
and `rico_held` is limited to n=3.
