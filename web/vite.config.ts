import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies `/api` and `/ws` to the FastAPI backend on :8000 so the
// frontend can be served from the Vite dev server (default :5173) without CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
