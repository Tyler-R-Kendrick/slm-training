# E344 bounded low-LR visible-slot adaptation — 2026-07-17

E344 restarts E343's 5k-token visible-slot adaptation from E337 but lowers
learning rate tenfold, from 3e-4 to 3e-5. Training completed in 115.7s and the
honest four-suite evaluation in 56.9s, each under the hard 300-second cap.

Final weighted/broad NLL are 6.0674/6.1001 and loss AgentV passes 2/2. The
local unsynced checkpoint SHA is
`b2fe39324cb649f07b8d55208c9c7374116ed99aacff0c31f5f06a87e545b074`.
Weights differ from E343, but all 16 predictions and every aggregate metric
are identical: parse is 1.0 throughout; smoke/held/adversarial/OOD fidelity is
0.1111/0.1400/0.1458/0.2167; meaningful rate, component recall, and reward are
zero throughout. AgentV passes 0/4 with no execution errors. RICO was omitted.

**Verdict:** reject E344. A 10x lower adaptation rate does not preserve E341's
decode-only semantic signal or change bounded quality. Do not promote or sync.

