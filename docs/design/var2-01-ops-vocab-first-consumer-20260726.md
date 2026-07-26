# VAR2-01 (SLM-428): OPS_VOCAB first consumer + adequacy audit

- **Status:** wiring reachable; adequacy audit published
- **Claim class:** `measurement` (adequacy audit), `wiring` (matched-arm probe)
- generated_at: `2026-07-26T21:49:27Z`
- ops_vocab fingerprint: `e039ebe5ab2ea6d746bd0757d16f65e2768fb4c0b8f4f37aa42c3e32472ecb93`
- corpus content fingerprint: `07a33b539530403f0d9881ae01b26af37ba5abc766efe5e847e7fd4b60653380`

## Honesty scoping (read first)

This script proves the wiring is reachable and computes the real adequacy audit over a small, real, deterministic corpus. It is **not** a trained-model quality experiment: matching the issue's acceptance criteria to "off vs on" SFT arms at fixed tokens/updates/seed needs a preregistered training campaign at a scale outside this repository's `MAX_RUN_MINUTES=3` hard cap for one autonomous session. `ModelBuildConfig.encoder_ops_conditioning` stays `False` by default regardless of this report's outcome; see `experiment_campaign` for the preregistered follow-on that would run that comparison.

## Adequacy audit (the deliverable)

- ops registered in OPS_VOCAB: 20
- ops actually used by this corpus: 4
- **unused_ops** (16): `conversation.ast_edit`, `conversation.checkout_state`, `conversation.copy_state`, `conversation.redo`, `conversation.transaction_commit`, `conversation.undo`, `openui.contract_subtree`, `openui.duplicate_subtree`, `openui.expand_template`, `openui.move_node`, `openui.reorder_children`, `openui.reparent_node`, `openui.set_property`, `openui.unset_property`, `openui.unwrap_node`, `openui.wrap_node`
  - of which **structurally unreachable by this consumer's own design** (not a corpus-coverage gap -- see decision below): `conversation.ast_edit`, `conversation.transaction_commit`
- **unrepresentable_intents** (0): none
- **arity_mismatches** (0): none
- **family_confusions**: UNRUN_CONDITIONAL -- Distinguishing op pairs the encoder cannot separate requires a trained encoder; this audit is measurement-only (no training run). Left for the campaign's follow-on trained comparison.

## Wiring probe (off vs on)

- representable_fraction: 1.000
- all off-arm outputs unchanged (lever disabled): True
- all on-arm reserved tokens correctly layered (I13, no grammar collision): True

| example | representable | off unchanged | reserved tokens layered | op count |
| --- | --- | --- | --- | --- |
| 1698df4da7ba30e073298ff448d1f8861349ab36d1a437555dbfb95c3b5be42a | True | True | True | 2 |
| 16afa5ca0d5dfdfc098e3d249f14bdc1d5d1679db7b3e0d8c44957c8c7384287 | True | True | True | 2 |
| 1dc88fb5a69e89c6e04aa72f83b4d89edd0e391177dc5ee0091a3216c8c558f2 | True | True | True | 1 |
| 1efda894acf6229c1572f36a7161eb5e484f65f8a075a5ed911bfac3b50f4979 | True | True | True | 2 |
| 2bec76ca2994f8392d2dcc084e43eabd58e35a9196963276e22ebc3d223febe9 | True | True | True | 1 |
| 321779e37db52db2a0952df81b6ce5a9aa642a2b21f5cfe42d356bfbea3bc851 | True | True | True | 1 |
| 3720884f9c3e1e3c4a5d530d5a088073c57cd3dfcd896a5d979debe40af46809 | True | True | True | 1 |
| 3981270dd4e3dda5474e03fbf8bc0f84466671d92dfc84ee7cfb022001dada25 | True | True | True | 2 |
| 3dae8342312c682c71f37f287f0400d3f5aa65243929b58f3c11cfa8e194579a | True | True | True | 1 |
| 4085d9bfeea4ef860996a895035050833a26b13a60ad5d624f70664dc9bf99be | True | True | True | 1 |
| 4d1aec2965cfeeea5ba59c02642bfc87ceb727b07d77974d09bcc541367edfbe | True | True | True | 2 |
| 6a00146694918c5b5c7a2d1150f267fbf9f9a26805e3f6624e333a8e4f07279d | True | True | True | 2 |
| 994af1438888cbf66c14cb03d04ef09d7c44cd088f0f7a96849365ade100ea5a | True | True | True | 1 |
| a21dfeddccf9d84da746c5f2acf9246f0356a3c676963c4111e287ae9cff260d | True | True | True | 1 |
| a8966df97938c10150dc22c5c72378a5026cf8c24cd97f3028b29a125cb67029 | True | True | True | 1 |
| bc2d734e8e064133a284f0d4bda7546cdff352149946c6154dcb4ca952b67856 | True | True | True | 1 |
| c4fb8bf5e8c599b201e65b5b21908ea391b543d9691f57c20f74116463606421 | True | True | True | 1 |
| db205e971b66044a9f4aac2cf3be1bddba55c4b53f26c9c9edf9bb0f09165e29 | True | True | True | 1 |
| e57b9135db031040fcacd8eac75b27aa83a48d7e611b6fac1e794d6ef195c494 | True | True | True | 1 |
| eb6f5ac8375b112c5931346fd4b2df3e5cf66e6e3644917a3044de7f25298830 | True | True | True | 1 |

## Decision this audit licenses

Wiring is reachable and non-regressing: 100% of audited turns resolve to a real reserved OPS_VOCAB token, and the off arm is byte-identical with the lever disabled. The adequacy audit finds 16/20 registered ops with zero occurrences in this corpus (see 'unused_ops') -- expected, since this corpus only exercises the local operator library; VAR2-02 (widening the corpus to topology/history operators, or accepting the narrower scope) is the named follow-on. Of those, 2 ('conversation.ast_edit'/'conversation.transaction_commit') are structurally unreachable by this consumer's own design at any corpus size -- an AST_EDIT/TRANSACTION_COMMIT turn always resolves to its specific underlying operator id, never to that literal history-family id -- so widening the corpus alone cannot close this part; it is a real kernel-shape finding, not a coverage gap. family_confusions is UNRUN_CONDITIONAL: no trained encoder exists yet to measure it. encoder_ops_conditioning remains default-off; turning it on for a real training run needs its own preregistered campaign, per this campaign's stopping rules.
