# OpenUI visual preview island

Bundles `@openuidev/react-lang` `Renderer` + `openuiLibrary` for the TwoTower annotate playground.

```bash
cd src/apps/openui_preview
npm ci
npm run build
# writes src/slm_training/web/static/preview/preview.js (+ CSS)
```

The playground loads `/static/preview/preview.js` and calls:

```js
window.OpenUIPreview.mount(el, { source: openuiLang });
```

Dataset captures use `capture.mjs` through
`slm_training.data.render.capture_program`. The capture adapter renders the
viewport × theme × data-state matrix, writes fixed/full-page/overlapping-tile
screenshots, and emits stable ProgramSpec statement/layout metadata.
Callers may provide per-state OpenUI sources when empty/loading/populated/error
are distinct programs; otherwise the same canonical ProgramSpec is captured
under each state label.
