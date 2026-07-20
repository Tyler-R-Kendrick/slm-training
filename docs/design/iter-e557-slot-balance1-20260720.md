# E557 — full slot-owner class balancing

E557 changes only E555's class-balance power from 0.5 to 1.0. It processed
1,304 target tokens in 70.09 seconds under `max_wall_minutes=3` and wrote SHA
`438d9871bc8389f6d61d4f3e357d56d280a22aafa26962404f6c47c92b97db05`.

OOD `n=4` predictions and every headline metric exactly match E555: fidelity
0.3000, structure 0.1594, recall 0.1250, reward 0.5453, AST node F1 0.2389,
meaning 0, and AgentV 0/1.

**Verdict:** reject the checkpoint and close scalar class-balance tuning.
Change owner coverage in data or sampling. Evidence:
[JSON](iter-e557-slot-balance1-20260720.json).
