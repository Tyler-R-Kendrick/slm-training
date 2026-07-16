# Iteration: post-fix seed-3 constrained feedback (2026-07-15)

Seed 3 used the matched scratch TwoTower recipe: eight steps, batch size `8`,
learning rate `6e-4`, effective batch size `8`, and the same 585-record corpus.
The run consumed **10,057** target tokens and its complete deterministic loss
suite reported weighted held-out NLL **30.709**.

After the constrained fallback probe fix, a one-record, one-step, one-attempt
constrained smoke evaluation completed in **2,753.24 ms** and persisted the
scoreboard plus AgentV JSONL. It recorded zero decode timeouts, but parse rate,
structural similarity, and reward were all **0**. This separates the prior
termination bug from model quality: the decoder now returns bounded feedback,
but the checkpoint still cannot produce valid OpenUI.

Decision: retain the corpus and recipe; use the now-terminating constrained
path for the next harness/model experiment. This is a scratch diagnostic, not a
ship claim.
