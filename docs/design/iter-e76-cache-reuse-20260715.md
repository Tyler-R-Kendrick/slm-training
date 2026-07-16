# E76 successor-cache reuse — 2026-07-15

E76 kept the E75 curriculum and V7 cluster/survival recipe while removing the
trust and entropy remask gates. The expected intervention was to make the
successor cache observable and reduce denoiser work.

The run did not exercise the cache: speculative batches, canvases, hits, and
misses were all zero in smoke and held-out telemetry. Quality was also below
the gate because placeholder fidelity was 0.0 (parse was 1.0/0.6 and
structure was 0.6111/0.4951 for smoke/held-out).

Decision: reject E76. The result exposes a harness/configuration gap: the
runtime received cluster activity but no successor speculation. The next
iteration must persist effective V7 decode flags and an explicit abstention
reason before making a throughput claim.

This is scratch smoke/held-out evidence, not a ship claim.
