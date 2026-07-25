# SLM-288 meaningful-program rubric (v1)

You are labeling UI-layout programs (OpenUI Lang) for **semantic
meaningfulness**: does this output constitute a genuine attempt at the
requested layout, rather than a trivial, empty, or off-task shell?

Label each item exactly one of:

- `meaningful` — a real attempt: parses, has populated containers with
  content components, binds the requested data placeholders, and covers the
  component types the prompt asks for.
- `not_meaningful` — trivial or off-task: unparseable, empty
  containers (e.g. `Stack([])` / `Card([])`), no content components, no
  placeholders, or clearly missing the requested component inventory.
- `unknown` — you cannot determine meaningfulness from the prompt and
  output shown (e.g. output appears truncated). Unknown is never counted as
  a failure or a success.

Also record any applicable reason codes:
`parse_failed`, `free_form_output_string`, `empty_root_stack`, `empty_card`,
`empty_children`, `no_content_components`, `no_placeholders`,
`low_component_recall`, `component_recall_unobservable`.

You are blinded: model/checkpoint identity is intentionally withheld. Judge
only the prompt and output shown. Do not look up repository artifacts.
