# Iteration: four-seed recipe variance (2026-07-15)

Four four-step scratch TwoTower runs used the remediated corpus, batch size 8,
learning rate `6e-4`, and final-only loss feedback. Weighted held-out NLLs:

- seed 0: 27.977
- seed 1: 31.165
- seed 2: 30.283
- seed 3: 31.102

Mean NLL is **30.132**, sample standard deviation **1.491**, and range
27.977–31.165. Every run persisted complete loss reports, run insights, and
telemetry with effective batch size 8. The bounded spread supports recipe
variance, not stable hard-example evidence; no data reweighting, deletion, or
recipe change is justified.
