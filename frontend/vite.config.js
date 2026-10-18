import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import vuetify from "vite-plugin-vuetify";

// The plugin page is served from a token-scoped sub path, so every emitted
// asset URL must stay relative.
export default defineConfig({
  base: "./",
  plugins: [vue(), vuetify({ autoImport: true })],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "../pages/dashboard",
    emptyOutDir: true,
    // Single page bundle: one JS chunk keeps the sandboxed iframe simple.
    chunkSizeWarningLimit: 1500,
  },
});
