# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **150** (latest `autotrain_wf_smoke_20260726_iter150`).

`iter146` onward rebuilds on `wf_smoke_v3` (fresh fixture corpus; `outputs/`
does not persist across containers) and carries the `harness.train_data` v21
canonical-template-marker fix (`canonicalize_example_template_markers` +
`assert_canonical_template_markers` in `_normalize_record`, cherry-picked from
the unmerged canonicalization fix) so the fresh build does not regress into
the earlier `canonical_template_marker_gate` failure. `iter147`-`iter150`
reuse the already-built `wf_smoke_v3` train/eval dirs (`data_train`/`data_test`
skipped, artifacts present), so `last_loss` is identical run-to-run
(deterministic seed=0/cpu over unchanged data) — expected, not a regression.

| run_id | ok | steps | stopped_on | last_loss | wall_s | max_wall |
| --- | --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725` | True | 8 | steps | 41.52362060546875 | 55.16 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter2` | True | 8 | steps | 36.6005744934082 | 46.78 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter3` | True | 8 | steps | 24.184091567993164 | 44.7 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter4` | True | 8 | steps | 38.17085647583008 | 48.27 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter5` | True | 8 | steps | 44.06878662109375 | 47.84 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter6` | True | 8 | steps | 29.073862075805664 | 42.78 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter7` | True | 8 | steps | 26.87131690979004 | 43.78 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter8` | True | 8 | steps | 34.350059509277344 | 46.46 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter9` | True | 8 | steps | 39.343326568603516 | 57.68 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter10` | True | 8 | steps | 38.7589111328125 | 48.11 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter11` | True | 8 | steps | 30.61425018310547 | 45.57 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter12` | True | 8 | steps | 44.137203216552734 | 35.22 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter13` | True | 8 | steps | 34.19657516479492 | 46.4 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter14` | True | 8 | steps | 26.997821807861328 | 40.42 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter15` | True | 8 | steps | 37.53594207763672 | 42.18 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter16` | True | 8 | steps | 33.343055725097656 | 40.55 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter17` | True | 8 | steps | 34.90048599243164 | 40.8 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter18` | True | 8 | steps | 35.47141647338867 | 39.51 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter19` | True | 8 | steps | 28.82457733154297 | 47.46 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter20` | True | 8 | steps | 37.124794006347656 | 35.2 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter21` | True | 8 | steps | 32.02244186401367 | 47.61 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter22` | True | 8 | steps | 33.407711029052734 | 53.29 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter23` | True | 8 | steps | 32.79945373535156 | 73.46 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter24` | True | 8 | steps | 30.783294677734375 | 67.27 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter25` | True | 8 | steps | 34.37016296386719 | 64.69 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter26` | True | 8 | steps | 27.447492599487305 | 54.96 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter27` | True | 8 | steps | 32.47339630126953 | 64.06 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter28` | True | 8 | steps | 28.06317138671875 | 69.04 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter29` | True | 8 | steps | 22.009885787963867 | 63.19 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter30` | True | 8 | steps | 29.3861083984375 | 68.48 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter31` | True | 8 | steps | 30.14482879638672 | 59.65 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter32` | True | 8 | steps | 29.287851333618164 | 58.45 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter33` | True | 8 | steps | 32.747520446777344 | 69.42 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter34` | True | 8 | steps | 35.80878448486328 | 56.76 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter35` | True | 8 | steps | 30.297271728515625 | 58.88 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter36` | True | 8 | steps | 32.42424392700195 | 63.49 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter37` | True | 8 | steps | 42.312870025634766 | 66.18 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter38` | True | 8 | steps | 20.78548812866211 | 62.97 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter39` | True | 8 | steps | 32.567710876464844 | 64.55 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter40` | True | 8 | steps | 31.621179580688477 | 69.92 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter41` | True | 8 | steps | 27.94664764404297 | 64.39 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter42` | True | 8 | steps | 33.96420669555664 | 72.57 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter43` | True | 8 | steps | 29.189895629882812 | 61.01 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter44` | True | 8 | steps | 34.544864654541016 | 56.38 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter45` | True | 8 | steps | 24.131763458251953 | 62.72 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter46` | True | 8 | steps | 30.152284622192383 | 67.76 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter47` | True | 8 | steps | 28.20587158203125 | 70.89 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter48` | True | 8 | steps | 23.86540985107422 | 64.7 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter49` | True | 8 | steps | 39.785682678222656 | 62.59 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter50` | True | 8 | steps | 28.041854858398438 | 80.08 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter51` | True | 8 | steps | 38.57737350463867 | 65.36 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter52` | True | 8 | steps | 37.48728561401367 | 62.02 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter53` | True | 8 | steps | 34.903202056884766 | 59.59 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter54` | True | 8 | steps | 22.906925201416016 | 72.76 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter55` | True | 8 | steps | 23.704858779907227 | 68.27 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter56` | True | 8 | steps | 36.08429718017578 | 66.22 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter57` | True | 8 | steps | 38.5702018737793 | 76.71 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter58` | True | 8 | steps | 19.11671257019043 | 69.86 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter59` | True | 8 | steps | 37.77717971801758 | 74.79 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter60` | True | 8 | steps | 28.253128051757812 | 69.16 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter61` | True | 8 | steps | 29.652488708496094 | 70.94 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter62` | True | 8 | steps | 29.386829376220703 | 77.44 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter63` | True | 8 | steps | 47.96480941772461 | 80.11 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter64` | True | 8 | steps | 25.109817504882812 | 86.61 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter65` | True | 8 | steps | 26.1611385345459 | 84.26 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter66` | True | 8 | steps | 34.344825744628906 | 70.66 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter67` | True | 8 | steps | 28.215469360351562 | 81.9 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter68` | True | 8 | steps | 33.62152099609375 | 71.01 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter69` | True | 8 | steps | 38.76063919067383 | 79.63 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter70` | True | 8 | steps | 28.972190856933594 | 86.98 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter71` | True | 8 | steps | 36.201175689697266 | 83.81 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter72` | True | 8 | steps | 34.49907684326172 | 63.85 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter73` | True | 8 | steps | 34.02790069580078 | 50.81 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter74` | True | 8 | steps | 31.445680618286133 | 49.33 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter75` | True | 8 | steps | 27.639373779296875 | 47.81 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter76` | True | 8 | steps | 29.802757263183594 | 45.99 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter77` | True | 8 | steps | 31.479494094848633 | 47.49 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter78` | True | 8 | steps | 24.801549911499023 | 46.32 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter79` | True | 8 | steps | 38.72297668457031 | 47.7 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter80` | True | 8 | steps | 34.16733932495117 | 46.53 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter81` | True | 8 | steps | 32.685611724853516 | 46.92 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter82` | True | 8 | steps | 28.008630752563477 | 56.68 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter83` | True | 8 | steps | 34.69600296020508 | 55.3 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter84` | True | 8 | steps | 37.947105407714844 | 61.22 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter85` | True | 8 | steps | 31.49188804626465 | 58.87 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter86` | True | 8 | steps | 24.548978805541992 | 61.87 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter87` | True | 8 | steps | 36.863731384277344 | 59.86 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter88` | True | 8 | steps | 28.695941925048828 | 68.12 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter89` | True | 8 | steps | 33.209228515625 | 44.7 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter90` | True | 8 | steps | 24.062515258789062 | 61.01 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter91` | True | 8 | steps | 29.29267120361328 | 48.06 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter92` | True | 8 | steps | 24.612091064453125 | 63.33 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter93` | True | 8 | steps | 18.60668182373047 | 52.34 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter94` | True | 8 | steps | 31.899532318115234 | 60.47 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter95` | True | 8 | steps | 43.52022933959961 | 52.79 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter96` | True | 8 | steps | 34.90167236328125 | 51.5 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter97` | True | 8 | steps | 29.899879455566406 | 51.98 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter98` | True | 8 | steps | 35.36892318725586 | 62.89 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter99` | True | 8 | steps | 26.564958572387695 | 55.9 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter100` | True | 8 | steps | 34.9030876159668 | 61.24 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter101` | True | 8 | steps | 32.23518371582031 | 60.89 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter102` | True | 8 | steps | 35.48078155517578 | 55.73 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter103` | True | 8 | steps | 31.048444747924805 | 55.51 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter104` | True | 8 | steps | 30.406570434570312 | 62.84 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter105` | True | 8 | steps | 36.59089660644531 | 53.23 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter106` | True | 8 | steps | 28.345420837402344 | 48.46 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter107` | True | 8 | steps | 32.68856430053711 | 55.74 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter108` | True | 8 | steps | 29.23394775390625 | 65.54 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter109` | True | 8 | steps | 33.150272369384766 | 55.51 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter110` | True | 8 | steps | 40.704532623291016 | 55.31 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter111` | True | 8 | steps | 25.30638313293457 | 54.99 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter112` | True | 8 | steps | 39.71058654785156 | 54.33 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter113` | True | 8 | steps | 35.18284225463867 | 50.14 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter114` | True | 8 | steps | 30.84190559387207 | 58.65 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter115` | True | 8 | steps | 28.262773513793945 | 57.78 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter116` | True | 8 | steps | 23.53805923461914 | 57.19 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter117` | True | 8 | steps | 28.04547882080078 | 54.88 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter118` | True | 8 | steps | 28.879148483276367 | 60.13 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter119` | True | 8 | steps | 41.54469680786133 | 57.06 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter120` | True | 8 | steps | 34.804710388183594 | 59.66 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter121` | True | 8 | steps | 27.98356056213379 | 57.44 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter122` | True | 8 | steps | 38.232994079589844 | 55.62 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter123` | True | 8 | steps | 32.90201187133789 | 61.16 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter124` | True | 8 | steps | 37.53460693359375 | 54.67 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter125` | True | 8 | steps | 34.56198501586914 | 63.41 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter126` | True | 8 | steps | 41.50794982910156 | 61.94 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter127` | True | 8 | steps | 25.588512420654297 | 58.67 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter128` | True | 8 | steps | 34.37063980102539 | 68.96 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter129` | True | 8 | steps | 27.891056060791016 | 56.21 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter130` | True | 8 | steps | 35.9130859375 | 60.84 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter131` | True | 8 | steps | 28.704561233520508 | 59.09 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter132` | True | 8 | steps | 30.90557098388672 | 53.18 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter133` | True | 8 | steps | 31.88162612915039 | 59.64 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter134` | True | 8 | steps | 41.917755126953125 | 57.11 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter135` | True | 8 | steps | 39.63530731201172 | 60.69 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter136` | True | 8 | steps | 33.11048889160156 | 52.49 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter137` | True | 8 | steps | 29.764385223388672 | 61.7 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter138` | True | 8 | steps | 31.387615203857422 | 57.78 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter139` | True | 8 | steps | 37.23638916015625 | 61.0 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter140` | True | 8 | steps | 34.07845687866211 | 58.5 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter141` | True | 8 | steps | 36.869117736816406 | 60.91 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter142` | True | 8 | steps | 34.74534606933594 | 59.05 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter143` | True | 8 | steps | 22.095991134643555 | 59.52 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter144` | True | 8 | steps | 34.957889556884766 | 58.55 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260725_iter145` | True | 8 | steps | 39.63420867919922 | 58.18 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260726_iter146` | True | 8 | steps | 30.91221809387207 | 61.48 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260726_iter147` | True | 8 | steps | 30.91221809387207 | 48.82 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260726_iter148` | True | 8 | steps | 30.91221809387207 | 48.26 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260726_iter149` | True | 8 | steps | 30.91221809387207 | 49.49 | 2.5833333333333335 |
| `autotrain_wf_smoke_20260726_iter150` | True | 8 | steps | 30.91221809387207 | 49.08 | 2.5833333333333335 |
