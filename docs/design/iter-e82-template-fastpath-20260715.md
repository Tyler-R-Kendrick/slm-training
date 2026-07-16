# E82 contract-template fast path — 2026-07-15

E82 trained the visible-contract corpus and enabled the new opt-in
`contract_template_fastpath`. When a serving request supplies a slot contract,
the decoder returns the certified contract-bound skeleton instead of entering
the expensive denoising/repair loop.

Bounded request-aware smoke evaluation (n=1) scored parse 1.0, exact
placeholder fidelity 1.0, contract precision/recall 1.0/1.0, structural
similarity 0.65, and reward 0.997 at 2,609.58 ms p50 with zero timeouts. The
training matrix probe used 64 steps; held-out, adversarial, OOD, and RICO
suites were not run.

Full bounded scoreboard with the request-aware fast path was: smoke parse /
fidelity / structure 1.0 / 1.0 / 0.6489; held-out 0.6 / 1.0 / 0.5539;
adversarial 1.0 / 1.0 / 0.8263; OOD 1.0 / 1.0 / 0.5974; and RICO held 1.0 /
1.0 / 0.7618. All suites had zero decode timeouts; p50 latency was 872.7 ms
smoke and 22.15–36.54 ms on the other suites.

Decision: retain the fast path as an opt-in harness lever, but do not promote
it as model quality: it returns a certified skeleton and therefore measures
contract plumbing/latency upper bound rather than learned semantic generation.

This is bounded scratch evidence, not a ship claim.
