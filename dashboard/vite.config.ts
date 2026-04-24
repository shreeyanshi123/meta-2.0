import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/reset': 'http://localhost:7860',
      '/step': 'http://localhost:7860',
      '/state': 'http://localhost:7860',
      '/health': 'http://localhost:7860',
      '/info': 'http://localhost:7860',
      '/stream': 'http://localhost:7860',
    },
  },
  build: {
    outDir: 'dist',
  },
})
