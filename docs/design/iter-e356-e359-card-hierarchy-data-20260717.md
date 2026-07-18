# E356–E359 RICO diagnosis and Card-hierarchy data — 2026-07-17

E356 evaluates RICO rows 128–191 in 68.4 seconds. Parse is 1.0, meaningful
rate 0.8594, fidelity 0.2932, structure 0.2879, component recall 0.4688, and
reward 0.6415. Nine examples fail meaningfulness due to low component recall;
AgentV correctly remains 0/1 because the 64-row artifact is diagnostic.

A train/eval audit over E353, E354, and E356 found one dominant structural
failure: `Card` occurs in 179/192 gold programs and is recovered 0/179 times.
The E316 training corpus contains `Card` in only 80/795 records. This is a
general structural coverage imbalance, not an eval-literal or one-record case.

E357 adds a deterministic train-only `card_hierarchy` synthesizer. For eligible
root `Stack` programs without an existing Card, it wraps each primary section
in its own Card and adds matching Card-count evidence to the prompt. The
immutable snapshot contains 998 records, including 203 accepted hierarchy
variants. Card occurrences rise from 80 to 435; 183 invalid or judge-sensitive
variants fail the existing G11 verifier and are excluded. Build errors and
reserved-test structure admissions are zero. Content fingerprint:
`a4f212a3444d0f219fe1b3604f70929fe1a1b91d4fdc11a73167cb74c55b6a51`.

E357's first data diagnostic correctly exposed that the 256-token decode canvas
cannot cover full RICO: 231/1500 targets exceed it and p95 is 280. E358 then
showed that `diagnose_eval --ltr-max-tokens 320` was silently capped by default
progressive stages ending at 256. The harness now appends an explicitly larger
maximum to the stages. E359 reruns at an effective 320 tokens: train p95/max is
96/112, RICO p95/max is 280/280, no rows exceed budget, all suite component
type and occurrence coverage is 1.0, and diagnostic AgentV passes 1/1.

All four commands completed under the hard 300-second cap.

**Verdict:** accept the E357 corpus for a matched bounded train and retain the
320-token diagnostic fix. This is data readiness evidence, not checkpoint
quality or a ship claim.
