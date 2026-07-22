# E844-E849: repeated typed collections

## Data builds

E844 rebuilt all current local sources under unchanged strict gates. It admitted
606/1,703 candidates and restored six repeated-TabItem rows that were absent
from the derived E826 snapshot, but still had no repeated FormControl or
SwitchItem rows. E845 added two canonical, three-item producer fixtures and
admitted 613/1,713 candidates. Repeated FormControl coverage became 2 rows
(maximum 3 instances) and repeated SwitchItem coverage became 4 rows (maximum
3); both source fixtures were retained. No gate was relaxed.

The required synthesis feedback was reviewed. E845 rejected 1,108 candidates:
803 dedup, 153 verification quarantine, 59 quality, 53 normalization, and 40
decontamination. Existing recommendations remain: reduce redundant template
and RICO expansion, repair quarantined Awwwards/scope producers, and audit
families with eval-overlap drops. E845 is retained as a candidate snapshot but
does not replace E826 as the default because its trained result regressed.

## Local train and smoke

E847 exposed a harness-default failure before training: the nominally local
command attempted an HF tokenizer lookup. It wrote no checkpoint or summary
and is not evidence. The centralized default is now `scratch`; HF context must
be selected explicitly.

E848 then completed 600 CPU steps on E845 in 69.88 seconds under the 95-second
harness cap. It wrote local checkpoint SHA
`036d6487c6d3b1c07b4515417495fa6a5e7673755c3fbaca4d900f841f35bed9`
with explicit `--no-sync-checkpoints`. E849 smoke `n=3` regressed versus E843:
parse/meaning-v1/strict-v2/fidelity 0.6667, structure 0.5300, component recall
0.5000, reward 0.6327, and one timeout. AgentV remained 0/1.

Reject E848 and do not promote E845. The next data arm should supplement E826
with the two canonical structural fixtures rather than replacing its curated
mixture with the broad all-source corpus. No remote workflow, bucket sync,
deployment, or ship claim occurred.
