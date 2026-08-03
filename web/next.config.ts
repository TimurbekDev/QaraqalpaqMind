import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // The browser never talks to the model gateway directly, so no rewrites or
  // CORS entries here. See app/api/chat/route.ts for why.
  poweredByHeader: false,
  // Emits .next/standalone with a self-contained server and only the packages
  // actually imported, so the runtime image needs no node_modules copy. Without
  // it the image carries the full dependency tree, most of which is build-time.
  output: "standalone",
};

export default config;
