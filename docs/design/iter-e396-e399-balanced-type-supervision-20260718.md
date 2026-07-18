# E396–E399 balanced type supervision — 2026-07-18

E393 isolated the remaining bounded-suite failure to held-out component type
recall: all target types occur in the immutable E357 corpus, but rare correct
types such as `Form`, `Tabs`, `TabItem`, `SwitchItem`, and `Slider` remained
below the component-plan decoder's activation threshold. Increasing
inventory or slot decode weights after training did not change any output.

E396 therefore resumes E368's full state on the same 998-record E357 corpus
and changes only existing supervision: component-plan loss weight 1→4 and
inverse-frequency slot-owner balancing power 0→0.5. The frozen SmolLM2 CPU
run reaches its 22,000-token budget after 104.6 seconds, 427 cumulative steps,
and 22,044 target tokens. Its local checkpoint SHA is
`feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`.
It has no new loss-suite result, inherits best weighted NLL 5.8091, is
explicitly not synced, and is not promoted.

E397 evaluates the complete five-row held-out suite with E393's unchanged
decode policy. Relative to E393, meaningful rate rises 0.4→0.6, structure
0.5061→0.5933, component recall 0.2333→0.4833, and reward 0.3922→0.5916.
Parse and placeholder fidelity remain 1.0.

E398 evaluates all four complete bounded suites. Every suite clears its local
quality thresholds and AgentV passes 4/4 with zero execution errors. The
global ship result remains false solely because full `rico_held` is absent.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.75 | 1.0 | 0.6762 | 0.7500 | 0.7268 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9827 |

E399 repeats E392's exact RICO rows 336–384. Parse, meaningful rate, and
fidelity remain 1.0 with zero row failures. Structure changes
0.6514→0.6401 and component recall 0.9931→0.8993, while reward stays 0.9991.
This is a bounded regression on an already-strong diagnostic shard, although
all shard-level quality floors remain clear. The shard is only 48/1500 and
cannot establish production readiness.

Every shell command used an external 290-second interrupt plus a forced kill
ten seconds later. E396 additionally used the trainer's internal 4.5-minute
wall limit. E396 stopped on its token budget; E397–E399 all completed normally.
Immediate setup failures loaded no model or evaluation rows and are excluded
from evidence.

**Verdict:** retain E396 as the strongest bounded HF-context candidate because
it fixes the held-out type-recall gate and preserves 4/4 bounded AgentV. Do not
promote or claim ship: the checkpoint is local and unsynced, no fresh NLL was
measured, full RICO is missing, and the matched RICO-48 shard regresses
structure and recall modestly.
