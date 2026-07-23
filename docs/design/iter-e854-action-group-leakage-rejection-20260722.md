# E854: action-group producer rejection

E853's weakest smoke row omitted the `Buttons` wrapper. A training-distribution
audit found that all 15 admitted `Buttons` examples were form-related, so E854
tested a generalized standalone single-action-group producer fixture derived
from E851 under unchanged strict gates.

The test-structure decontamination gate rejected the fixture before admission:
its topology matched a committed smoke structure. The resulting 351-row corpus
had the same content fingerprint as E851
(`b5193be89e42fb052a056584f22a9159cf1dccbb1edf2271ec1941cd41e56fb9`).
Synthesis feedback correctly identified the `human_curated` source and emitted
an eval-leakage recommendation. The fixture was removed; no train, checkpoint,
evaluation, remote workflow, or ship claim followed.

The next valid data arm must source structurally distinct standalone action
groups from real or programmatic producers rather than hand-authoring the held
out topology or weakening decontamination.
