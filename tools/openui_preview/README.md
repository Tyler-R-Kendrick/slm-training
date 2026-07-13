# OpenUI visual preview island

Bundles `@openuidev/react-lang` `Renderer` + `openuiLibrary` for the TwoTower annotate playground.

```bash
cd tools/openui_preview
npm ci
npm run build
# writes src/slm_training/web/static/preview/preview.js (+ CSS)
```

The playground loads `/static/preview/preview.js` and calls:

```js
window.OpenUIPreview.mount(el, { source: openuiLang });
```
