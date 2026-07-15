---
name: frontier-describe
description: Create or refresh committed train-only frontier description bundles for OpenUI gold records. Use when filling src/slm_training/resources/frontier/worklist.jsonl with paraphrases, L1-L5 abstraction prompts, minimal edit instructions, or optional external-page vision semantics while preserving placeholders, preventing DSL or literal-copy leakage, and rebuilding the deterministic coverage manifest.
---

# Frontier Describe

Produce teacher-authored natural-language metadata as frozen JSON. Never call a
model from the Python build and never describe test or held-out records.

## Workflow

1. From the repository root, refresh the train-only worklist:

   ```bash
   python -m scripts.frontier_worklist
   ```

2. Read `src/slm_training/resources/frontier/SCHEMA.md`, then process only worklist rows whose
   `has_fresh_artifact` is false. Do not open test or held-out fixtures.
3. For each row, create
   `src/slm_training/resources/frontier/<gold_id>.<gold_content_hash>.json`:

   - copy `gold_id`, `gold_content_hash`, and `skeleton_openui` byte-for-byte;
   - stamp `provenance.skill = "frontier-describe"`, `skill_version = "0.1.0"`,
     `prompt_hash = "sha256:<raw-prompt-sha256>"`, and an ISO-8601 `generated_at`;
   - write distinct placeholder-safe `paraphrases`;
   - write exactly `L1` through `L5` under `ladder`, ordered from semantic graph
     to vague intent;
   - write minimal `edits` as `edit_op`, natural-language `instruction`, and a
     stable `delta_ref`; copy a verified P4 delta reference when one exists;
   - write `vision.semantic_description` only when the worklist source includes
     approved external-page evidence.

4. Never invent or rewrite OpenUI. Never place OpenUI assignments/component
   calls in generated text. Keep content references as existing `:namespace.slot`
   placeholders; never introduce a new placeholder or replace one with literal
   copy. Do not reuse the source prompt verbatim.
5. Validate bundles and rebuild coverage:

   ```bash
   python .agents/skills/frontier-describe/scripts/finalize.py
   python -m scripts.frontier_worklist
   python .agents/skills/frontier-describe/scripts/finalize.py
   ```

   Fix every reported invalid or leakage finding before commit. Pending rows are
   allowed for a seed/partial pass; use `--require-complete` when the requested
   worklist must be fully filled.
6. Re-run the same commands. A clean rerun must leave current bundles and
   `MANIFEST.json` unchanged and list only genuinely missing/stale rows.

## Boundaries

- Commit JSON artifacts, the worklist, and `MANIFEST.json`; do not add runtime
  LLM SDKs, network calls, or generated OpenUI.
- Treat the exact hash-named path as current. Ignore stale bundles rather than
  mutating their provenance to match a changed gold.
- Stop if the copied skeleton/hash does not match the worklist; regenerate the
  worklist instead of guessing.
