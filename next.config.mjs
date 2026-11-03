import path from 'path'
import { fileURLToPath } from 'url'

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Silences the "multiple lockfiles" warning: pin the workspace root to this project.
  outputFileTracingRoot: path.dirname(fileURLToPath(import.meta.url)),
}

export default nextConfig
