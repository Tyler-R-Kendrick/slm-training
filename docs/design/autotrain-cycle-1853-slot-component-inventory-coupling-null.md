# Autotrain c1853: slot-component/inventory coupling

**Verdict:** null and capacity-negative fixture result; rejected.

The hierarchical slot/inventory treatment did not recover any guarded quality
signal. Candidate and control both produced zero MPR, recall, binder F1,
fidelity, reward, and exact AST/canonical matches. The added inventory head
raised parameters by `77,916` (`+4.84%`) and increased p50 latency by `5.39%`.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware control | 1,608,962 | 9.3509 | .115 | 0 | 0 | 0 | 0 | 856 | 27 | 2 |
| slot-component + inventory | 1,686,878 | 11.7884 | .115 | 0 | 0 | 0 | 0 | 902 | 27 | 2 |

Smoke `n=3` is insufficient and all production suites are absent. This arm
therefore demonstrates no learning signal and confirms that stacking auxiliary
heads can buy capacity without buying capability. Keep the matched control,
prioritize target/data coverage, and require parameter-efficiency evidence for
any future auxiliary head.

Machine evidence:
[`autotrain-cycle-1853-slot-component-inventory-coupling-null.json`](autotrain-cycle-1853-slot-component-inventory-coupling-null.json).
