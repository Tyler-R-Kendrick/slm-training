---
name: dashboard-openui-parity
description: Use whenever a dashboard page changes — editing any tools/dashboard/src/pages/*.tsx or the shared components in tools/dashboard/src/components.tsx, or when adding/removing a route in main.tsx. The dashboard renders every page two ways (a compiled/interpreted toggle): hand-written React (compiled) and a committed OpenUI Lang program run live through the official @openuidev Renderer (interpreted). The two must stay at parity. This skill is the loop for updating the matching src/slm_training/web/static/openui/<slug>.openui program, the interpreted-mode library/toolProvider, and re-validating. Also use when a .openui program, tools/dashboard/src/interpret/*, or the page manifest is edited.
---

# Dashboard OpenUI parity (keep compiled ↔ interpreted in sync)

The dashboard ships **two renderers for the same pages**, switchable at runtime by
the sidebar **◈ Compiled / ◇ Interpreted** toggle (`localStorage "slm-mode"`,
`data-mode` on `:root`):

- **compiled** — the hand-written React pages under `tools/dashboard/src/pages/*.tsx`.
- **interpreted** — the committed **OpenUI Lang** programs under
  `src/slm_training/web/static/openui/<slug>.openui`, fetched and run live through
  the official `@openuidev/react-lang` `<Renderer>` with the dashboard's hybrid
  component library and `/api` tool provider (`tools/dashboard/src/interpret/`).

**They must render the same thing.** When you change a page, update its `.openui`
program too, or the two drift and interpreted mode is wrong.

## Interpreted-mode moving parts

| File | Role |
| --- | --- |
| `src/slm_training/web/static/openui/<slug>.openui` | The page program (one per route). Full OpenUI Lang: `Query`, `@Each`, `$state`, ternaries, custom components. |
| `tools/dashboard/src/interpret/library.tsx` | Hybrid component library: stock `@openuidev` components **+** `defineComponent` wrappers of the dashboard's real React widgets (StatTile, DataTable, GateMatrix, JobConsole, …). Wrappers guarantee pixel parity. |
| `tools/dashboard/src/interpret/toolProvider.ts` | Maps each `Query("name", …)` to `/api`, reshaping responses into DSL-friendly row-sets (the interpreted analogue of a compiled page's `usePoll`). Pre-format numeric cells to strings so table precision matches compiled. |
| `tools/dashboard/src/interpret/{DslView,actions,nav}.tsx` | The renderer host, action handler, and in-app nav ref. |
| `scripts/validate_page_dsl.py` | Structural validator + `MANIFEST.json` writer (see below). |

**Dialect note:** these programs are **not** the placeholder-only training DSL that
`tools/openui_bridge` validates. They use the full OpenUI Lang + the dashboard
library, so validate with `scripts/validate_page_dsl.py`, **never** the training
`validate`/bridge. Training data and ship-gates are untouched by this.

## The parity loop

For a page (`overview`, `data`, `experiments`, `smoke`, `checkpoints`, `playground`):

1. **Read the compiled page** (`tools/dashboard/src/pages/<Page>.tsx`) — note the exact
   structure: page-head, tile grid (and its `min` width), each card's title + right-slot
   badge/chip/link, tables (columns, per-column number precision, status pills), bars,
   and any interactive widgets (selectors, launchers, gate editor).
2. **Write / edit** `src/slm_training/web/static/openui/<slug>.openui` to reproduce it.
   Prefer real DSL (`Query` + `@Each` + components); reactive selectors bind a
   `$var` via `ChipTabs($var, [...])` so `Query({ arg: $var }, …)` refetches on click.
   Wrap genuinely stateful/imperative widgets (job launchers, live gate editor, the
   annotate playground) as custom components in `library.tsx` — that is the "Hybrid"
   contract and what yields parity.
3. **Add any missing** `toolProvider` query or `library` component the program needs.
4. **Build**: `env -u NODE_OPTIONS npm --prefix tools/dashboard run build`
   (the sandbox sets `NODE_OPTIONS="--import tsx"`, which breaks `node`/`npm` — always
   prefix node/npm with `env -u NODE_OPTIONS`). The `.openui` files are served verbatim
   from `/static/openui/`, so DSL-only edits need **no** rebuild — only library /
   toolProvider changes do.
5. **Diff both renders.** Serve (`python -m scripts.serve_playground`), then load the
   page in each mode and compare a structural DOM fingerprint (page-title, card titles +
   right slots, tile label/value/sub/accent, bar counts, table rows/heads, pills, empties)
   plus a screenshot. Toggle mode by seeding `localStorage "slm-mode"` before load.
   Iterate step 2–5 until the fingerprint matches and the screenshots are identical.
6. **Validate + manifest**: `python scripts/validate_page_dsl.py` (rewrites
   `static/openui/MANIFEST.json`). Commit the manifest.

Parity gotchas learned the hard way:

- Custom `defineComponent` renderers receive `{ props, renderNode, statementId }` —
  props are **nested under `.props`**, and child nodes are `ElementNode` descriptors
  you render via `renderNode(...)` (not spread).
- Compiled distinguishes a **missing** `sub` (no `.tile-sub` div) from `sub=""` (empty
  div). Emit `null` for "no sub", `""` only where compiled passes `""`.
- Reactive `$state` from a **custom** component only reaches `Query` args when the
  prop is declared `reactive(z...)` and the program passes the `$var` itself (mirror how
  stock inputs use `useStateField(name, props.value)`).

## Keep-in-sync (enforced)

- `scripts/validate_page_dsl.py` — every `Query("x")` resolves to a `toolProvider` key,
  every `Component(...)` to a `library.tsx`/stock component, brackets balance, each route
  has a program; rewrites `MANIFEST.json`. Run it after any edit.
- `tests/test_web/test_page_dsl.py` — the pytest guard (routes covered, references real,
  manifest fresh).
- `.claude/settings.json` `PostToolUse` hook runs the validator (`--changed`) on every
  Edit/Write/MultiEdit.
- CI runs `python scripts/validate_page_dsl.py --check` after `pytest -q`.

If you add or remove a route in `main.tsx`, add/remove the matching `.openui` program
(or add the slug to `COMPILED_ONLY_SLUGS` in the validator if it is intentionally
compiled-only, like the dynamic `/runs/:id` detail view).
