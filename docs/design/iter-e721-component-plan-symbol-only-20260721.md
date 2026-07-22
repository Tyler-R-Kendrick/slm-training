# E721 symbol-only role/count component-plan diagnostic

**Date:** 2026-07-21  
**Decision:** reject the checkpoint and decode lever; retain syntax evidence only  
**Evidence:** [`iter-e721-component-plan-symbol-only-20260721.json`](iter-e721-component-plan-symbol-only-20260721.json)

## Question

E720's validly bounded predictions often ran into incomplete or overgrown
programs. E721 tests the existing generalized grammar-role component plan: a
prompt-conditioned root class plus remaining bound-component counts, applied
only among compiler-legal component candidates. It adds no strings, literals,
component cases, or remote dependencies.

## Bounded local recipe

- CPU scratch TwoTower; exact 141-record output-contract-v2 snapshot, manifest
  `78096191b5272658a1f6c48b6cdceaba5492f363630c0bf7d1b19d2f465b2b45`.
- Lexer output, grammar-LTR primary, honest slot contract, tree compiler,
  component-plan loss/decode weights 1.0, 160-symbol canvas.
- Three requested budgets (600, 420, 280) reached only steps 245, 241, and 241
  before the 110-second interrupt. They are invalid attempts and contribute no
  metrics or checkpoints to the decision.
- The measured local ceiling motivated a 190-step recipe. It terminated in
  90.39 seconds under cumulative `max_wall_minutes=2`, seeing 97,891 prompt and
  20,342 target tokens. Compiler-plan work dominated forward time (76.94%).
- Local-only checkpoint SHA
  `c30fd565fced08626f39af5e9e23d233d88c26e0dac3b031928105b97c20f530`;
  explicit `--no-sync-checkpoints`, not promoted.

At step 190, total loss was 7.3057 and primary reconstruction loss was 4.4437.
The plan head learned the root target (accuracy 1.0), but bound top-k recall was
0.25 and bound-count MAE 0.2608.

## Matched smoke evaluation

Both arms use three frozen smoke prompts, one attempt, no unconstrained
fallback, an eight-second per-example timeout, and AgentV.

| Decode arm | Parse | Strict-v2 | Fidelity | Structure | Recall | Reward | Timeouts | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plan weight 1 | 1.0000 | 0.0000 | 0.5278 | 0.1513 | 0.3333 | 0.0000 | 0/3 | 4938.06 / 5183.59 ms |
| Plan weight 0 | 1.0000 | 0.0000 | 0.5278 | 0.1513 | 0.3333 | 0.0000 | 0/3 | 5036.39 / 5047.05 ms |

The treatment recorded 39 plan applications but zero choice changes. Aggregate
metrics and predictions match the plan-off control. All three outputs parse,
but all fail strict meaning as trivial duplicate-heavy layouts; AgentV is 0/1
per arm with zero execution errors.

## Disposition

E721 is useful syntax evidence but not evidence for the component-plan decoder.
The 1.0 parse rate is non-causal with respect to the decode weight, strict-v2
meaning remains 0.0, and the supervision path is too expensive for more than
about 240 local steps inside the hard cap. Reject the checkpoint and lever; do
not upload, promote, or ship. The next arm must target semantic duplicate and
required-component failures without another plan-weight or duration sweep.
