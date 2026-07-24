# E1088: dependency-closed root-tail pairs

E1088 adds one deterministic layout augmentation: when a generated document has
at least three root `Stack` siblings, retain its final two siblings and the
transitive declaration closure. It re-derives the target-only opaque slot
contract from the projected program. The transform uses neither prompt marker
meanings nor held-out records.

The strict all-source rebuild completed locally within the canonical cap and
published immutable `e1088_root_tail_pair_v1`. It admits 532 rows, eight more
than E937's 524; layout augmentation admits 105 of 256 candidates (0.4102)
with independent-judge and runtime verification both 1.0. Decontamination
still rejects 28 layout candidates and flags all leakage rather than weakening
any threshold.

The candidate most similar to the held Settings failure,
`train_settings_01_aug_tail2`, was rejected at decontamination as
`test_fixture_structure`. It is absent from E1088, so it is not a valid
training explanation for the held row. The next fresh local scratch run may
test only the eight structurally disjoint additions. This snapshot is
train-only diagnostic evidence: it has no checkpoint, promotion, bucket sync,
serving, or ship claim.

E1089 is invalidated before evaluation. Its first bounded invocation stopped
at 207/395 steps, but the next invocation omitted the required
`--resume-from` flag and restarted, overwriting that partial state with a new
83-step run. Neither artifact is a completed checkpoint or evidence. E1090
will start fresh and explicitly resume its own `last_full_state.pt` until it
reaches exactly 395 steps.

That valid E1090 chain reached exactly 395 steps, but its matched E1091
Settings evaluation regressed from E1087's non-timeout parse (fidelity 0.3333,
reward 0.707) to a 12.01-second timeout and empty prediction (all those metrics
0). Reject E1088 as a positive intervention. Its leakage guard remains valid,
but neither E1090 nor this snapshot may support promotion, serving, sync, or
parent selection. See `iter-e1090-root-tail-pair-train-20260724.md`.
