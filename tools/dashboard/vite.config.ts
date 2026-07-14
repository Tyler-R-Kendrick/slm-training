import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Build the SPA into the FastAPI static tree (committed, like the preview lib).
// `base` makes emitted asset URLs absolute under /static/app/ so the shell can be
// served from "/" and any deep client route.
export default defineConfig({
  plugins: [react()],
  base: "/static/app/",
  build: {
    outDir: path.resolve(__dirname, "../../src/slm_training/web/static/app"),
    emptyOutDir: true,
    sourcemap: true,
    target: "es2022",
  },
});
