import path from "node:path";
import type { NextConfig } from "next";

// A static export: the whole lab runs in the browser, so it can be hosted anywhere.
const basePath = process.env.BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  images: { unoptimized: true },
  transpilePackages: ["tessera-engine", "tessera-lab"],
  turbopack: { root: path.join(__dirname, "../..") },
};

export default nextConfig;
