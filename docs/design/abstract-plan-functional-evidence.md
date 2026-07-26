# SLM-313 AbstractPlan functional evidence

This locked functional-evidence record makes no model or promotion claim.

- Verdict: ignored_or_collapsed; promotion eligible: false.
- Reason: All pre-locked local shards were present and merged with exact arm/path coverage.
- Locked manifest: b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48.
- Meaningful-parse status: measured_full_locked_matrix.
- AgentEvals/AgentV records the fail-closed non-promotion assertion.

## Complete local result

CPU scratch Choice TwoTower checkpoint `slm313_local_plan_1k_v2` (9 steps,
1,006 target tokens, 24.70s; explicit no-sync) ran every locked record across
9 arms and 3 decode paths: 6,102 rows total. The primary meaningful-v2 and
binder/reference-F1 deltas versus empty, random-norm-matched, and shuffled
plans are all 0.0 with paired 95% CIs `[0.0, 0.0]`. The learned connector adds
56.55ms mean latency and 1.14% total tokens versus no-plan. This rejects the
learned-plan functional claim as **ignored_or_collapsed**; the checkpoint is
diagnostic-only and neither reusable, promotable, nor ship.

## Execution history

  - 2026-07-26T06:34:28.351038+00:00: invalid_local_run — Locked matrix did not merge: ValueError: incomplete locked shard coverage: missing 225 shard(s)
  - 2026-07-26T06:49:15.265896+00:00: invalid_local_run — Local shard did not complete: KeyError: 'rico_eval_test_2188'
  - 2026-07-26T06:56:14.570718+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T06:57:50.099622+00:00: invalid_local_run — Local shard did not complete: TimeoutError: SLM-313 shuffle_between_examples/repaired shard 0 timed out; numeric evidence is invalid
  - 2026-07-26T07:02:07.259680+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:04:49.281169+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:05:59.403624+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:08:04.010986+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:10:02.021038+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:12:02.392334+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:13:38.366271+00:00: invalid_local_run — Local shard did not complete: TimeoutError: SLM-313 length_matched_verbal/repaired shard 5 timed out; numeric evidence is invalid
  - 2026-07-26T07:17:53.840056+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:19:47.759123+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:21:43.411670+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:23:38.810572+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:26:21.523846+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:28:23.573603+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:30:23.028467+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:32:29.823002+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:34:19.074760+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:36:13.886407+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:38:13.176749+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:40:11.393954+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:42:08.312989+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:44:03.745870+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:46:02.478675+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:47:51.870429+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:49:59.149072+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:51:54.528917+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:53:52.725861+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:55:52.610422+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T07:57:55.495974+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:00:01.524524+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:02:02.806702+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:03:59.556960+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:06:01.957102+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:08:00.835470+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:09:58.646637+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:12:00.125995+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:14:02.101485+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:16:06.627458+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:18:19.017285+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:20:23.802806+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:22:27.913898+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:24:24.255315+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:26:22.705179+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:28:20.457597+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:30:11.562144+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:32:13.459255+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:34:26.140725+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:36:22.819661+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:38:17.833407+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:40:04.212873+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:41:40.452646+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:43:18.365912+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:44:57.767339+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:46:34.414134+00:00: partial_locked_shard — A complete shard is measured; final disposition requires exact merge coverage across every locked shard.
  - 2026-07-26T08:47:04.039396+00:00: ignored_or_collapsed — All pre-locked local shards were present and merged with exact arm/path coverage.
