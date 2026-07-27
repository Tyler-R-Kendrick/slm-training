# Abstract-CoT warm-up orchestration (SLM-311 / AP-021)

SLM-311 provides a bounded, resumable T-iteration coordinator for the existing
Abstract-CoT path. It does not introduce a decoder, collector, or trainer:

- AP-020's collector remains the sole on-policy trace collector and preserves
  its prompt/answer duplicate and resume guarantees.
- AP-019's training payload builder remains the sole producer of the
  segment-masked SFT input.
- AbstractWarmupCampaignV1 locks the iteration manifest before any work, and
  CampaignStore records immutable per-iteration artifacts/events.

Each completed iteration has an addressable policy reference, its parent policy,
the exact command/configuration/input/output hashes, token and update counts,
and one evidence record for each AP-020 collection and AP-019 segment-masked
SFT hand-off. The configured default is T=3.

Resume is event-based. A completed event is reused; an iteration directory
without that event is left untouched and causes a fail-closed error. This
prevents duplicate collection and checkpoint/policy overwrites after an
interruption. No locked test is read or used for iteration selection.

## Scope and evidence

The local fixture mode is a hermetic wiring proof. It invokes the real
collector with a fixture verifier and builds the real AP-019 tensor payload,
but it does not instantiate or train a model. Its JSON, Markdown, and
AgentEvals/AgentV artifacts explicitly record:

- no model checkpoint was written;
- meaningful-parse was not measured;
- no locked suite was used for selection; and
- the result is not promotion- or ship-eligible.

A future production caller must bind a real AP-017 abstract-traced policy and
apply the returned AP-019 payload through the existing segment-masked training
entry point. It must then run the preregistered campaign/evaluation process and
cannot promote based on this fixture evidence.
