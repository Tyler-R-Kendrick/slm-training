# A02: reject execution grants absent from the scientific lock

This focused fix closes the `cmd_run` admission gap: a campaign with a 600-second
logical grant previously reached compilation when its manifest omitted that
grant or declared 601 seconds. Both explicit-file and previously locked manifest
paths now reject this mismatch before compilation, experiment start, or work.
The full ResourceGrant must match; a successor manifest/campaign is required to
change it. Legacy absent-grant JSON is unchanged, and historical 60-minute wall
fields still permit at most 180-second invocations (the executor keeps its
170-second interrupt cap). No scientific threshold or evaluator changes.

The unchanged invocation-budget calculation is extracted into CampaignBudget;
the public script helpers remain compatible. This offsets the admission guards
without raising the script's 2,114-line quality ceiling.

Local verification used the preserved candidate, Python 3.12, and:

```text
timeout -s INT -k 10 170 env PYTHONPATH=src:outputs/runs/autonomy-validation/dependencies OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/codex/repos/slm-training/.venv/bin/python -m pytest -q -o addopts= -m '' --tb=short --show-capture=no --junitxml=outputs/runs/autonomy-validation/a02-publish-grant-lock.xml tests/test_autoresearch/test_campaign_grant_lock.py tests/test_autoresearch/test_logical_grant_binding.py
```

Result: 21 passed, exit 0, 2.09 seconds. Nine tests are the focused guard/legacy
suite delivered here. Twelve additional local tests cover the broader A02 WIP;
they are not evidence that all of A02 is delivered by this patch. The original
broader test run reproduced four failures (two producer omissions and two
admission failures). Receipts and SHA256 values are in the accompanying JSON.

Publication is a narrow patch against remote main
`a809e81878302baa27a9d6fedd8f0b79ee3af857`, not a wholesale upload of the dirty
candidate. Remote main lacks `scripts/autotrain_search.py`; its producer fixes
and treatment-resource identity changes remain local/integration obligations.
Full exact-PR-tree release verification and integration-owner version accounting
remain outstanding. No model training, service start, hosted checks, promotion,
or live repair execution was performed for this fix.
