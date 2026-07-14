# Frozen frontier artifacts — contract (F3 / SLM-4)

The **frontier work is an agent-skill**, not a runtime API. The `frontier-describe`
skill (P5) reads a worklist of train golds and writes committed JSON bundles here;
the deterministic Python build **never calls a model** — it only reads these
bundles, binds each to the exact gold, and **re-validates every row** it emits.
This keeps `build_train_data` reproducible (`content_fingerprint` stable).

## Layout

```
fixtures/frontier/
  worklist.jsonl                 # scripts/frontier_worklist.py — TRAIN golds only
  <gold_id>.<gold_hash8>.json    # one bundle per gold, keyed by content hash
```

- `gold_hash8 = gold_content_hash(openui, prompt)` — `sha256(strip_style_literals(openui) ⊕ prompt)[:16]`
  (`slm_training.data.frontier.gold_content_hash`). If a gold changes, its hash
  changes, the filename no longer resolves, and the stale bundle is **dropped**
  until the skill regenerates it.

## `worklist.jsonl` row

```json
{"gold_id": "train_hero_01", "gold_content_hash": "ab12cd34ef56ab78",
 "prompt": "...", "skeleton_openui": "root = Stack([...])\n...",
 "has_fresh_artifact": false}
```

## Bundle `<gold_id>.<gold_hash8>.json`

Skeleton-only: every target stays placeholderized (`:ns.slot`); the skill never
emits literal copy or OpenUI it invents.

```json
{
  "gold_id": "train_hero_01",
  "gold_content_hash": "ab12cd34ef56ab78",
  "skeleton_openui": "root = Stack([...])\n...",      // must be structurally == the gold
  "provenance": {"skill": "frontier-describe", "skill_version": "0.1.0",
                 "prompt_hash": "sha256:...", "generated_at": "2026-07-14T00:00:00Z"},
  "paraphrases": ["...", "..."],                       // NL prompts, target unchanged
  "ladder": {"semantic": "...", "product": "...",      // canonical L1/L3/L4/L5 aliases
             "user": "...", "simplified": "..."},
  "edits": [{"edit_op": "add", "instruction": "...", "after_openui": "..."}],  // read by P4
  "vision": {"semantic_description": "...", "screenshot_path": "..."}          // read by P7
}
```

## Consumption

`FrozenArtifactSynthesizer` (`harnesses/train_data/synth.py`, name `frontier`)
implements the `PromptSynthesizer.expand(record)` protocol:

1. compute `gold_content_hash(record)`, load `<id>.<hash>.json` (miss → no rows);
2. **faithfulness bind** — reject unless `structure_fp(skeleton_openui) == structure_fp(gold)`;
3. emit one row per `paraphrase` (family `frontier_described`) and per `ladder`
   rung (`frontier_semantic` / `frontier_product` / `frontier_user` /
   `frontier_simplified`), keeping the verified skeleton as the target and
   `task="generation"`; ladder rows carry canonical `L0`–`L5`, fact contracts,
   constraint coverage, target determinacy, and any house-style resolution;
4. the pipeline re-validates + decontaminates every emitted row as usual.

`edits` / `vision` are defined here but consumed by later stages (edit oracle P4,
vision bridge P7). **Only `split == "train"` golds are ever described.**
