import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: path.resolve(__dirname, "../../src/slm_training/web/static/preview"),
    emptyOutDir: true,
    lib: {
      entry: path.resolve(__dirname, "src/main.tsx"),
      name: "OpenUIPreview",
      formats: ["es"],
      fileName: () => "preview.js",
    },
    rollupOptions: {
      output: {
        assetFileNames: "preview.[ext]",
      },
    },
    cssCodeSplit: false,
    sourcemap: true,
    target: "es2022",
  },
});
