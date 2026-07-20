# E591 — content-property owner slot score

Date: 2026-07-20
Status: positive diagnostic; not promotable or ship

E591 adds a default-off score for visible slots whose prompt-derived component
role matches the active schema-marked content property owner. It changes no
legal candidates and uses only visible prompt/slot evidence.

All arms use CPU, frozen local HF context, honest visible slot/role contracts,
constrained LTR, 8 steps, 4 attempts, and a 160-token canvas. Every process
completed under 170 seconds. On the matched E590 OOD `n=4` recipe, weights 2
and 4 are byte- and metric-identical. Relative to weight 0, fidelity improves 0.4250→0.5917,
validity 0.6550→0.7550, reward 0.7585→0.8085, AST-node F1
0.4889→0.5198, and AST-edge F1 0.2500→0.3250. Structure is nearly flat
(0.4069→0.4044). Auth becomes one Input with name/email plus one Button,
instead of two Inputs with the create slot incorrectly used as a placeholder.

Use weight 2 as the next scratch baseline. Do not promote or sync: strict
meaning-v2 remains zero, AgentV is 0/1, and the subset has four records.

Evidence: [JSON](iter-e591-property-role-slot-20260720.json).
