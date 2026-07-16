# E103 repair-heavy mixture (2026-07-15)

E103 increased judged `corruption_repair` sampling to 45% and reduced prompt
paraphrases to 10%, using a committed diagnostic mixture manifest. The corpus
remained 1,417 records; the 128-step CPU run saw 37,587 target tokens and
finished at loss `6.32719`. Training telemetry is persisted in the run
directory.

Strict smoke evaluation produced raw syntax validity `1.0`, but parse `0.0`,
contract precision/recall `0.5/0.25`, placeholder fidelity `0.25`, structural
similarity `0.1917`, and component recall `0.0`. Latency was `5395.91 ms`;
fallback count was zero. AgentV remained non-ship.

Decision: reject the repair-heavy mixture. It improves surface syntax only by
losing contract/content fidelity, so repair records need better selection or
quality filtering rather than simply higher sampling weight.
