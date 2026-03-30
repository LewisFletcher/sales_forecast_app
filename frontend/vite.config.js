import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
  preview: {
    allowedHosts: 'front-end-production-6da9.up.railway.app',
  },
})
