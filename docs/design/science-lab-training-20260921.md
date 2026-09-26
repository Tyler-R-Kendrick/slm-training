# SCI-LAB paired training diagnostic

The finite CPU scratch run compared two size-matched TwoTower arms on six
public evaluation cases. Both used seed 7301, 65,826 trainable parameters and
six logical updates. Each continued exactly from the same three-update prefix;
uninterrupted references matched the resumed checkpoints. The control used
learning rate 0.0003 and the candidate 0.0006. Training data contained eight
records. Checkpoints remain local under
`outputs/autonomy-integration-20260921/science-lab-outputs/`; they were not
synced or promoted.

The ordinary supervisor completed both arms, decoded all six rows per arm,
produced AgentEvals/AgentV artifacts, and recorded a complete paired diagnostic.
Conditional masked-token cross-entropy was 24.234531 for control and 22.469211
for candidate (six non-tied pairs; exact two-sided sign-test p=0.03125).
Parse rate was 1.0 for both; meaningful-program rate was 1/6 for both. The
primary diagnostic improved, but quality bars failed and the result is not an
independent confirmation, a promotion, or a shipping claim.

This measurement is historical evidence from a dirty checkout based on the
version-stamped commit below. It proves the ordinary-supervisor path for that
captured source identity only; it does not certify the integration changes or
their merged source. The result and recipe are retained for continuity, not
reused as current-source acceptance.

## Version stamp

`docs/design/science-lab-training-20260921.json` carries the run's
`version_stamp/v1` component identities.
