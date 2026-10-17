/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Бэкенд (A02) в dev-режиме: VITE_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
const target = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  // Относительный base: dist можно раздавать с "/" или из подкаталога.
  // Роутинг — через hash (#/job/<id>), поэтому HTML всегда один.
  base: "./",
  plugins: [react()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    // Шрифты/иконки — в бандле, без CDN (офлайн-контур банка).
    assetsInlineLimit: 4096,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ui": { target, changeOrigin: true },
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
