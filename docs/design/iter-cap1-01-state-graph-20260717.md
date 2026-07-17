# CAP1-01 bounded OpenUI state-graph analysis

Generated: 2026-07-17T23:36:05.624201+00:00Z

## Constraint frame

```json
{
  "allowed_component_subset": [
    "Card",
    "TextContent",
    "Button",
    "Stack"
  ],
  "dsl": "openui",
  "max_components": 2,
  "max_list_items": 3,
  "max_literal_slots": 1,
  "max_live_bindings": 3,
  "max_object_members": 3,
  "max_semantic_decisions": 6,
  "profile_id": "openui-cap-v1",
  "representation": "choice",
  "required_coverage": "complete"
}
```

## Result summary

- Status: **EXACT**
- Exact: yes
- Raw states: 171
- Minimized states: 26
- Transitions: 13333
- Terminal: 88
- Invalid: 0
- Unknown: 0

## Work counters

```json
{
  "edges_explored": 13333,
  "forced_tokens_collapsed": 11918,
  "pruned_actions": 0,
  "states_seen": 171
}
```

## Histograms

- Branching: {1: 80, 3: 4, 4: 6, 7: 4, 84: 12, 85: 2, 95: 2, 96: 2, 153: 2, 154: 18, 155: 6, 173: 8, 174: 4, 175: 5, 285: 1, 286: 8, 287: 5, 290: 1, 368: 1}
- Forced suffix length: {0: 4223, 1: 6302, 2: 2808}

## Honest caveat

This is a bounded structural quotient over the choice-codec owner. It is wiring evidence only and does not claim that the minimized count is the latent model capacity or ship-grade state optimum.
