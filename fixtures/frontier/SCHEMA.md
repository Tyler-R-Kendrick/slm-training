# Frozen frontier artifact schema

The frontier skill reads `worklist.jsonl` and writes one committed bundle named
`<gold_id>.<gold_content_hash[:8]>.json`. Python never calls an LLM at build time.

Each bundle is a JSON object with:

- `schema_version`: `1`
- `gold_id`, `gold_content_hash`, and `skeleton_openui`
- `provenance`: non-empty `skill_name`, `skill_version`, `prompt_hash`, and ISO-8601 `generated_at`
- `paraphrases`: prompt strings
- `ladder`: objects with `prompt`, `level` (`L1`–`L5`), and `determinacy`
- `edits`: objects with `prompt`, optional `openui`, and optional `task` (default `edit`)
- `vision`: objects with `prompt` and any evidence references needed by the skill

All four blocks are required arrays, even when empty. The reader ignores a bundle
unless its gold hash is current and `skeleton_openui` has the same structural
fingerprint as the train gold. Test and held-out records are never worklisted or
expanded.
